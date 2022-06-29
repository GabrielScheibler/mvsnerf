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

create_mesh = True
is_finetuned = False
scene = 1

enum = enumerate([1,8,21,103,114])

if is_finetuned:
    enum = enumerate([scene])

for i_scene, scene in enum:#,8,21,103,114
    psnr,ssim,LPIPS_vgg,rgbs = [],[],[],[]
    cmd = f'--datadir /abyss/home/data/mvsnerf_dtu/mvs_training/dtu/scan{scene}  \
     --dataset_name dtu_ft  \
     --net_type neus --ckpt ./ckpts//latest.tar '

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

        if create_mesh:
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
                #volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
                imgs_source = unpreprocess(imgs_source)

                H, W = img.shape[:2]

                print(imgs_source.shape)

                img_source1 = torch.clamp(imgs_source[0,0].cpu(),0,1.0).numpy()
                img_source2 = torch.clamp(imgs_source[0,1].cpu(),0,1.0).numpy()
                img_source3 = torch.clamp(imgs_source[0,2].cpu(),0,1.0).numpy()
                


                img_source1 = np.transpose(img_source1, (1,2,0))*255
                img_source2 = np.transpose(img_source2, (1,2,0))*255
                img_source3 = np.transpose(img_source3, (1,2,0))*255

                

                #img_vis = imgs_source*255
            
                if save_as_image:
                    imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}_1.png', img_source1.astype('uint8'))
                    imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}_2.png', img_source2.astype('uint8'))
                    imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}_3.png', img_source3.astype('uint8'))

               