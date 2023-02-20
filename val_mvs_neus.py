import sys,os,imageio,lpips,cv2,torch,glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

import sys,os,imageio,lpips
root = '/workspace/mvsnerf'
os.chdir(root)
sys.path.append(root)

from opt import config_parser
from data import dataset_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# models
from models import *
from renderer import *
from data.ray_utils import get_rays

from tqdm import tqdm


from skimage.metrics import structural_similarity

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers


from data.ray_utils import ray_marcher

# torch.cuda.set_device(2)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs

def unpreprocess(data, shape=(1,1,3,1,1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std

def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32) # (800, 800)
    depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_NEAREST)  # (600, 800)
    depth_h = depth_h[44:556, 80:720]  # (512, 640)
#     depth = cv2.resize(depth_h, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_NEAREST)#!!!!!!!!!!!!!!!!!!!!!!!!!
    mask = depth>0
    return depth_h,mask

loss_fn_vgg = lpips.LPIPS(net='vgg') 
mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)

def acc_threshold(abs_err, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    acc_mask = abs_err < threshold
    return  acc_mask.astype('float') if type(abs_err) is np.ndarray else acc_mask.float()

def mse(x,gt,mask=None):
    if mask==None:
        mask = np.ones_like(x).astype(bool)
    return np.mean((x[mask] - gt[mask])**2)

def mae(x,gt,mask=None):
    if mask==None:
        mask = np.ones_like(x).astype(bool)
    return np.mean(np.abs(x[mask] - gt[mask]))

psnr_everywhere_all,ssim_everywhere_all,LPIPS_everywhere_vgg_all = [],[],[]
psnr_in_all,ssim_in_all,LPIPS_in_vgg_all = [],[],[]
psnr_out_all,ssim_out_all,LPIPS_out_vgg_all = [],[],[]
mask_error_all = []

depth_acc = {}
eval_metric = [0.1,0.05,0.01]
depth_acc[f'abs_err'],depth_acc[f'acc_l_{eval_metric[0]}'],depth_acc[f'acc_l_{eval_metric[1]}'],depth_acc[f'acc_l_{eval_metric[2]}'],depth_acc[f'cast_abs_err'] = {},{},{},{},{}

is_finetuned = False
scene = 114

enum = enumerate([1,8,21,103,114])

if is_finetuned:
    enum = enumerate([scene])

for i_scene, scene in enum:#,8,21,103,114
    rgbs = []
    psnr_everywhere, ssim_everywhere,LPIPS_everywhere_vgg = [],[],[]
    psnr_in, ssim_in,LPIPS_in_vgg = [],[],[]
    psnr_out, ssim_out,LPIPS_out_vgg = [],[],[]
    cmd = f'--datadir /abyss/home/data/mvsnerf_dtu/mvs_training/dtu/scan{scene}  \
     --dataset_name dtu_ft  \
     --net_type neus --ckpt ./ckpts//latest.tar --with_depth --neus_sampling'

    args = config_parser(cmd.split())
    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim =  8+12

    # create models
    if 0==i_scene:
        render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(render_kwargs_train)

        pytorch_total_params_fn = sum(p.numel() for p in render_kwargs_train['network_fn'].parameters() if p.requires_grad)
        pytorch_total_params_mvs = sum(p.numel() for p in render_kwargs_train['network_mvs'].parameters() if p.requires_grad)

        MVSNet = render_kwargs_train['network_mvs']
        render_kwargs_train.pop('network_mvs')


    datadir = args.datadir
    datatype = 'train'
    pad = 24
    args.chunk = 5120


    print('============> rendering dataset <===================')
    dataset_train = dataset_dict[args.dataset_name](args, split='train')
    dataset_val = dataset_dict[args.dataset_name](args, split='val')
    val_idx = dataset_val.img_idx

    if 0==i_scene:
        print('params_fn: ', pytorch_total_params_fn)
        print('params_mvs: ', pytorch_total_params_mvs)
    
    save_as_image = True
    save_dir = f'results/test3'
    os.makedirs(save_dir, exist_ok=True)
    MVSNet.train()
    MVSNet = MVSNet.cuda()


    with torch.no_grad():

        try:
            tqdm._instances.clear() 
        except Exception:     
            pass

        for i, batch in enumerate(tqdm(dataset_val)):
            torch.cuda.empty_cache()
            
            rays, img = decode_batch(batch)
            rays = rays.squeeze().to(device)  # (H*W, 3)
            img = img.squeeze().cpu().numpy()  # (H, W, 3)
            depth = batch['depth'].squeeze().numpy()  # (H, W)
        
            # find nearest image idx from training views
            positions = dataset_train.poses[:,:3,3]
            dis = np.sum(np.abs(positions - dataset_val.poses[[i],:3,3]), axis=-1)
            pair_idx = np.argsort(dis)[:3]
            pair_idx = [dataset_train.img_idx[item] for item in pair_idx]
            
            imgs_source, proj_mats, near_far_source, pose_source = dataset_train.read_source_views(pair_idx=pair_idx,device=device)
            volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
            imgs_source = unpreprocess(imgs_source)
            depth_map_input = None
            if args.with_depth_map:
                depth_map_input = pose_source['depths'].unsqueeze(0)

            if is_finetuned:   
                imgs_source, proj_mats, near_far_source, pose_source = dataset_train.read_source_views(device=device)
                imgs_source = unpreprocess(imgs_source)
                volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
                volume_feature = RefVolume(volume_feature.detach()).cuda()
                pad = 24
        
            N_rays_all = rays.shape[0]
            rgb_fg_rays, rgb_bg_rays = [],[]
            rgb_rays, depth_rays_preds = [],[]
            cast_depth_rays_preds = []
            fg_mask_pred = []
            mask_error = []
            for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                    N_samples=args.N_samples)

                if('neus' in args.net_type and args.neus_sampling):
                    xyz_coarse_sampled, z_vals, _ = gen_pts_neus(
                        rays_o, rays_d, imgs_source, volume_feature, pose_source, near_far_source, args, False, render_kwargs_train["network_fn"], render_kwargs_train["network_query_fn"], depth_map=depth_map_input)



                # Converting world coordinate to ndc coordinate
                H, W = img.shape[:2]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)

                cos_anneal_ratio = 1.0

                # rendering
                rgb, disp, acc, depth_pred, alpha, extras, rgb_fg, rgb_bg, sdf_gradient_error = rendering(args, pose_source, xyz_coarse_sampled,
                                                                                                          xyz_NDC, z_vals, rays_o, rays_d, inv_scale, cos_anneal_ratio,
                                                                                                          volume_feature, imgs_source, depth_map=depth_map_input, **render_kwargs_train)
    
                #print(torch.mean(rgb_bg))
                fg_mask = extras['mask_fg']
                cast_depth_pred = extras['cast_depth_map'].cpu().numpy()

                rgb_fg, rgb_bg = torch.clamp(rgb_fg.cpu(),0,1.0).numpy(),torch.clamp(rgb_bg.cpu(),0,1.0).numpy()
                rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
                fg_mask = torch.clamp(fg_mask.cpu(),0,1.0).numpy()
                rgb_fg_rays.append(rgb_fg)
                rgb_bg_rays.append(rgb_bg)
                rgb_rays.append(rgb)
                depth_rays_preds.append(depth_pred)
                cast_depth_rays_preds.append(cast_depth_pred)
                fg_mask_pred.append(fg_mask)

            
            depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
            cast_depth_rays_preds = np.concatenate(cast_depth_rays_preds).reshape(H, W)
            fg_mask_pred = np.concatenate(fg_mask_pred).reshape(H, W)

            depth_gt, _ =  read_depth(f'/abyss/home/data/mvsnerf_dtu/mvs_training/dtu/Depths/scan{scene}/depth_map_{val_idx[i]:04d}.pfm')
        
            mask_gt = depth_gt>0
            abs_err = abs_error(depth_rays_preds, depth_gt/246.40544, mask_gt)
            cast_abs_err = abs_error(cast_depth_rays_preds, depth_gt/246.40544, mask_gt)

            eval_metric = [0.01,0.05, 0.1]
            depth_acc[f'abs_err'][f'{scene}'] = np.mean(abs_err)
            depth_acc[f'acc_l_{eval_metric[0]}'][f'{scene}'] = acc_threshold(abs_err,eval_metric[0]).mean()
            depth_acc[f'acc_l_{eval_metric[1]}'][f'{scene}'] = acc_threshold(abs_err,eval_metric[1]).mean()
            depth_acc[f'acc_l_{eval_metric[2]}'][f'{scene}'] = acc_threshold(abs_err,eval_metric[2]).mean()
            depth_acc[f'cast_abs_err'][f'{scene}'] = np.mean(cast_abs_err)

            
            depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
            
            rgb_fg_rays = np.concatenate(rgb_fg_rays).reshape(H, W, 3)
            rgb_bg_rays = np.concatenate(rgb_bg_rays).reshape(H, W, 3)
            rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
            img_vis = np.concatenate((img*255,rgb_rays*255,rgb_fg_rays*255,rgb_bg_rays*255,depth_rays_preds),axis=1)
            
            if save_as_image:
                imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}.png', img_vis.astype('uint8'))
            else:
                rgbs.append(img_vis.astype('uint8'))

            mask_error.append(mse(fg_mask_pred,mask_gt))
                
            #TODO add measurement how well fore and background are differentiated between? is it possible? is foreground defined?  (add depth and mask loss)
            # quantity
            # mask background since they are outside the far boundle
            mask = depth==0
            imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}_mask.png', mask.astype('uint8')*255)
            psnr_everywhere.append( mse2psnr(np.mean((rgb_rays-img)**2)))
            ssim_everywhere.append( structural_similarity(rgb_rays, img, multichannel=True))
            rgb_rays_in,img_in = np.copy(rgb_rays),np.copy(img)
            rgb_rays_in[mask],img_in[mask] = 0.0,0.0
            psnr_in.append( mse2psnr(np.mean((rgb_rays_in[~mask]-img_in[~mask])**2)))
            ssim_in.append( structural_similarity(rgb_rays_in, img_in, multichannel=True))
            rgb_rays_out,img_out = np.copy(rgb_rays),np.copy(img)
            rgb_rays_out[~mask],img_out[~mask] = 0.0,0.0
            psnr_out.append( mse2psnr(np.mean((rgb_rays_out[mask]-img_out[mask])**2)))
            ssim_out.append( structural_similarity(rgb_rays_out, img_out, multichannel=True))
            
            img_tensor = torch.from_numpy(rgb_rays)[None].permute(0,3,1,2).float()*2-1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
            img_gt_tensor = torch.from_numpy(img)[None].permute(0,3,1,2).float()*2-1.0
            LPIPS_everywhere_vgg.append( loss_fn_vgg(img_tensor, img_gt_tensor).item())
            img_tensor = torch.from_numpy(rgb_rays_in)[None].permute(0,3,1,2).float()*2-1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
            img_gt_tensor = torch.from_numpy(img_in)[None].permute(0,3,1,2).float()*2-1.0
            LPIPS_in_vgg.append( loss_fn_vgg(img_tensor, img_gt_tensor).item())
            img_tensor = torch.from_numpy(rgb_rays_out)[None].permute(0,3,1,2).float()*2-1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
            img_gt_tensor = torch.from_numpy(img_out)[None].permute(0,3,1,2).float()*2-1.0
            LPIPS_out_vgg.append( loss_fn_vgg(img_tensor, img_gt_tensor).item())

            

        print(f'=====> scene: {scene} mean psnr {np.mean(psnr_everywhere)} ssim: {np.mean(ssim_everywhere)} lpips: {np.mean(LPIPS_everywhere_vgg)}')   
        psnr_everywhere_all.append(psnr_everywhere);ssim_everywhere_all.append(ssim_everywhere);LPIPS_everywhere_vgg_all.append(LPIPS_everywhere_vgg)
        print(f'=====> scene: {scene} mean psnr in {np.mean(psnr_in)} ssim: {np.mean(ssim_in)} lpips: {np.mean(LPIPS_in_vgg)}')   
        psnr_in_all.append(psnr_in);ssim_in_all.append(ssim_in);LPIPS_in_vgg_all.append(LPIPS_in_vgg)
        print(f'=====> scene: {scene} mean psnr out {np.mean(psnr_out)} ssim: {np.mean(ssim_out)} lpips: {np.mean(LPIPS_out_vgg)}')   
        psnr_out_all.append(psnr_out);ssim_out_all.append(ssim_out);LPIPS_out_vgg_all.append(LPIPS_out_vgg)
        print(f'=====> scene: {scene} mean mask_error {np.mean(mask_error)}') 
        mask_error_all.append(mask_error)
        print(f'=====> scene: {scene} mean depth_error {np.mean(depth_acc[f"abs_err"][f"{scene}"])}')
        print(f'=====> scene: {scene} mean dcast_epth_error {np.mean(depth_acc[f"cast_abs_err"][f"{scene}"])}') 
        

    if not save_as_image:
        imageio.mimwrite(f'{save_dir}/{scene}_spiral.mp4', np.stack(rgbs), fps=20, quality=10)

print(f'============> ERROR <=================')
a = np.mean(list(depth_acc['abs_err'].values()))
b = np.mean(list(depth_acc[f'acc_l_{eval_metric[0]}'].values()))
c = np.mean(list(depth_acc[f'acc_l_{eval_metric[1]}'].values()))
d = np.mean(list(depth_acc[f'acc_l_{eval_metric[2]}'].values()))
e = np.mean(list(depth_acc['cast_abs_err'].values()))
print(f'============> abs_err: {a} <=================')
print(f'============> acc_l_{eval_metric[0]}: {b} <=================')
print(f'============> acc_l_{eval_metric[1]}: {c} <=================')
print(f'============> acc_l_{eval_metric[2]}: {d} <=================')
print(f'============> cast_abs_err: {e} <=================')
print(f'=====> all mean psnr_everywhere {np.mean(psnr_everywhere_all)} ssim: {np.mean(ssim_everywhere_all)} lpips: {np.mean(LPIPS_everywhere_vgg_all)}') 
print(f'=====> all mean psnr_in {np.mean(psnr_in_all)} ssim: {np.mean(ssim_in_all)} lpips: {np.mean(LPIPS_in_vgg_all)}') 
print(f'=====> all mean psnr_out {np.mean(psnr_out_all)} ssim: {np.mean(ssim_out_all)} lpips: {np.mean(LPIPS_out_vgg_all)}') 
print(f'=====> all mean mask_error {np.mean(mask_error_all)}') 

print(f'============> VARIANCE <=================')
a = np.var(list(depth_acc['abs_err'].values()))
b = np.var(list(depth_acc[f'acc_l_{eval_metric[0]}'].values()))
c = np.var(list(depth_acc[f'acc_l_{eval_metric[1]}'].values()))
d = np.var(list(depth_acc[f'acc_l_{eval_metric[2]}'].values()))
e = np.var(list(depth_acc['cast_abs_err'].values()))
print(f'============> var abs_err: {a} <=================')
print(f'============> var acc_l_{eval_metric[0]}: {b} <=================')
print(f'============> var acc_l_{eval_metric[1]}: {c} <=================')
print(f'============> var acc_l_{eval_metric[2]}: {d} <=================')
print(f'============> var cast_abs_err: {e} <=================')
print(f'=====> all var psnr_everywhere {np.var(psnr_everywhere_all)} ssim: {np.var(ssim_everywhere_all)} lpips: {np.var(LPIPS_everywhere_vgg_all)}') 
print(f'=====> all var psnr_in {np.var(psnr_in_all)} ssim: {np.var(ssim_in_all)} lpips: {np.var(LPIPS_in_vgg_all)}') 
print(f'=====> all var psnr_out {np.var(psnr_out_all)} ssim: {np.var(ssim_out_all)} lpips: {np.var(LPIPS_out_vgg_all)}')
print(f'=====> all var mask_error {np.var(mask_error_all)}') 