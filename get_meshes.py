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

psnr_all,ssim_all,LPIPS_vgg_all = [],[],[]
depth_acc = {}
eval_metric = [0.1,0.05,0.01]
depth_acc[f'abs_err'],depth_acc[f'acc_l_{eval_metric[0]}'],depth_acc[f'acc_l_{eval_metric[1]}'],depth_acc[f'acc_l_{eval_metric[2]}'] = {},{},{},{}

gen_alpha = True
gen_sdf = True
is_finetuned = False
scene = 114

#enum = enumerate([1,8,21,103,114])
enum = enumerate([114])

if is_finetuned:
    enum = enumerate([scene])

for i_scene, scene in enum:#,8,21,103,114
    psnr,ssim,LPIPS_vgg,rgbs = [],[],[],[]
    cmd = f'--datadir /abyss/home/data/mvsnerf_dtu/mvs_training/dtu/scan{scene}  \
     --dataset_name dtu_ft  \
     --net_type neus --ckpt ./ckpts//latest.tar --with_depth'

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
            depthmaps = pose_source['depths'].unsqueeze(0)
            volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
            imgs_source = unpreprocess(imgs_source)

            if is_finetuned:   
                imgs_source, proj_mats, near_far_source, pose_source = dataset_train.read_source_views(device=device)
                imgs_source = unpreprocess(imgs_source)
                volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
                volume_feature = RefVolume(volume_feature.detach()).cuda()
                pad = 24

            # create 3D mesh
            near = -1
            far = 1
            resolution = 200

            pts, rays_o, rays_d, z_vals = point_grid([near,near,near],[far,far,far],resolution)
            pts, rays_o, rays_d, z_vals = pts.to(device), rays_o.to(device), rays_d.to(device), z_vals.to(device)
            xyz_coarse_sampled = pts

            # Converting world coordinate to ndc coordinate
            H, W = img.shape[:2]
            inv_scale = torch.tensor([W - 1, H - 1]).to(device)
            w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
            xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                            near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)
            cos_anneal_ratio = 1.0

            depth_map_input = None
            if args.with_depth_map:
                depth_map_input = depthmaps

            if gen_alpha:
                print("\ngenerate models via alpha method:")

                # rendering
                rgb, disp, acc, depth_pred, alpha, extras, _, _, _ = rendering(args, pose_source, xyz_coarse_sampled,
                                                                        xyz_NDC, z_vals, rays_o, rays_d, inv_scale, cos_anneal_ratio,
                                                                        volume_feature,imgs_source,depth_map=depth_map_input, **render_kwargs_train)
                #sdf = 1 - alpha
                
                pts_norm = torch.linalg.norm(pts, ord=2, dim=-1)
                outside_sphere = (pts_norm > 1.0).bool()

                # outside_frame, _ = torch.max(xyz_NDC,dim=-1)
                # outside_frame = outside_frame > 1

                alpha[outside_sphere] = 0.0

                threshold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

                alpha = alpha.reshape([resolution,resolution,resolution])
                alpha = alpha.cpu().detach().numpy()
                for th in threshold:
                    print("threshhold: ",th)
                    print("inside ratio: ",np.sum(alpha>th) / np.sum(np.ones_like(alpha)))
                    validate_mesh(save_dir, 1-alpha, i_scene*10**4 + i*10**6 + 0*10**3 + round(th*10**2), world_space=False,
                                resolution=resolution, threshold=1-th)
                #3D mesh done
                print("done generating models via alpha method\n")

            if 'neus' in args.net_type and gen_sdf:
                print("\ngenerate models via sdf method:")

                # rendering
                sdf = mesh_rendering(args, pose_source, xyz_coarse_sampled,
                                        xyz_NDC, inv_scale,
                                        volume_feature, imgs_source, depth_map=depth_map_input, **render_kwargs_train)

                pts_norm = torch.linalg.norm(pts, ord=2, dim=-1)
                outside_sphere = (pts_norm > 1.0).bool()

                # outside_frame, _ = torch.max(xyz_NDC,dim=-1)
                # outside_frame = outside_frame > 1

                sdf[outside_sphere] = 1.0

                threshold = [0.0]

                sdf = sdf.reshape([resolution,resolution,resolution])
                sdf = sdf.cpu().detach().numpy()
                for th in threshold:
                    print("threshhold: ",th)
                    print("inside ratio: ",np.sum(sdf<th) / np.sum(np.ones_like(sdf)))
                    validate_mesh(save_dir, sdf, i_scene*10**4 + i*10**6 + 1*10**3 + round(th*10**2), world_space=False,
                                resolution=resolution, threshold=th)
                #3D mesh done
                print("done generating models via sdf method\n")

            break

        print("\nall 3D meshes done")