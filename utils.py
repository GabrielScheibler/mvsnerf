import os, torch, cv2, re
import numpy as np
import math


from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

from scipy.spatial.transform import Rotation as R

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
mse2psnr2 = lambda x : -10. * np.log(x) / np.log(10.)

def get_psnr(imgs_pred, imgs_gt):
    psnrs = []
    for (img,tar) in zip(imgs_pred,imgs_gt):
        psnrs.append(mse2psnr2(np.mean((img - tar.cpu().numpy())**2)))
    return np.array(psnrs)

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log



def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def abs_error_numpy(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return np.abs(depth_pred - depth_gt)

def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    err = depth_pred - depth_gt
    return np.abs(err) if type(depth_pred) is np.ndarray else err.abs()

def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.astype('float') if type(depth_pred) is np.ndarray else acc_mask.float()


# Ray helpers
def get_rays_mvs(H, W, intrinsic, c2w, N=1024, isRandom=True, is_precrop_iters=False, chunk=-1, idx=-1):

    device = c2w.device
    if isRandom:
        if is_precrop_iters and torch.rand((1,)) > 0.3:
            xs, ys = torch.randint(W//6, W-W//6, (N,)).float().to(device), torch.randint(H//6, H-H//6, (N,)).float().to(device)
        else:
            xs, ys = torch.randint(0,W,(N,)).float().to(device), torch.randint(0,H,(N,)).float().to(device)
    else:
        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        if chunk>0:
            ys, xs = ys[idx*chunk:(idx+1)*chunk], xs[idx*chunk:(idx+1)*chunk]
        ys, xs = ys.to(device), xs.to(device)

    dirs = torch.stack([(xs-intrinsic[0,2])/intrinsic[0,0], (ys-intrinsic[1,2])/intrinsic[1,1], torch.ones_like(xs)], -1) # use 1 instead of -1


    rays_d = dirs @ c2w[:3,:3].t() # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].clone()
    pixel_coordinates = torch.stack((ys,xs)) # row col
    return rays_o, rays_d, pixel_coordinates



def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
    '''
        point_samples [N_rays N_sample 3]
    '''

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    # wrap to ref view
    if w2c_ref is not None:
        R = w2c_ref[:3, :3]  # (3, 3)
        T = w2c_ref[:3, 3:]  # (3, 1)
        point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)

    if intrinsic_ref is not None:
        # using projection
        point_samples_pixel =  point_samples @ intrinsic_ref.t()
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  # normalize to 0~1
        if not lindisp:
            point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
        else:
            point_samples_pixel[:,2] = (1.0/point_samples_pixel[:,2]-1.0/near)/(1.0/far - 1.0/near)
    else:
        # using bounding box
        near, far = near.view(1,3), far.view(1,3)
        point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
    del point_samples

    if pad>0:
        W_feat, H_feat = (inv_scale+1)/4.0
        point_samples_pixel[:,1] = point_samples_pixel[:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
        point_samples_pixel[:,0] = point_samples_pixel[:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)

    point_samples_pixel = point_samples_pixel.view(N_rays, N_samples, 3)
    return point_samples_pixel

def inverse_get_ndc_coordinate(c2w_ref, intrinsic_ref, point_samples_pixel, inv_scale, near=2, far=6, pad=0, lindisp=False):
    '''
        point_samples [N_rays N_sample 3]
    '''

    N_rays, N_samples = point_samples_pixel.shape[:2]
    point_samples_pixel = point_samples_pixel.reshape(-1, 3)

    # print(point_samples)

    if pad>0:
        W_feat, H_feat = (inv_scale+1)/4.0
        point_samples_pixel[:,1] = (point_samples_pixel[:,1] - pad / (H_feat + pad * 2)) / H_feat * (H_feat + pad * 2) 
        point_samples_pixel[:,0] = (point_samples_pixel[:,0] - pad / (W_feat + pad * 2)) / W_feat * (W_feat + pad * 2) 

    if intrinsic_ref is not None:
        # using projection
        # point_samples_pixel = point_samples_pixel * [[1,1,(far - near)]] + [[0,0,near]]
        # eye = torch.eye(3)
        # eye[2,0] = -1
        # eye[1,0]
        # points_samples_pixel = point_samples_pixel * point_samples_pixel[:,-1:] @ [[1,0,0],[0,1,0],[0,0,1]]

        point_samples_pixel[:,2] = (point_samples_pixel[:,2]) * (far - near) + near  # normalize to near~far
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] * point_samples_pixel[:,-1:]) * inv_scale.reshape(1,2)  # normalize to W,H
        point_samples =  point_samples_pixel @ torch.tensor(np.linalg.inv(intrinsic_ref.cpu())).t().cuda()
    else:
        near, far = near.view(1,3), far.view(1,3)
        point_samples_pixel = point_samples_pixel * (far - near) + near  # normalize to 0~1

    # wrap to ref view
    if c2w_ref is not None:
        R = c2w_ref[:3, :3]  # (3, 3)
        T = c2w_ref[:3, 3:]  # (3, 1)
        point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)
        # print(point_samples)
        # print(torch.sum(point_samples * point_samples, dim=-1))

    point_samples = point_samples.view(N_rays, N_samples, 3)
    return point_samples

def build_rays(imgs, depths, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=0, is_precrop_iters=False, ref_idx=0, importanceSampling=False, with_depth=False, is_volume=False):
    '''

    Args:
        imgs: [N V C H W]
        depths: [N V H W]
        poses: w2c c2w intrinsic [N V 4 4] [B V levels 3 3)]
        init_depth_min: [B D H W]
        depth_interval:
        N_rays: int
        N_samples: same as D int
        level: int 0 == smalest
        near_fars: [B D 2]

    Returns:
        [3 N_rays N_samples]
    '''

    device = imgs.device

    N, V, C, H, W = imgs.shape
    w2c_ref, intrinsic_ref = pose_ref['w2cs'][ref_idx], pose_ref['intrinsics'][ref_idx]  # assume camera 0 is reference
    inv_scale = torch.tensor([W-1, H-1]).to(device)

    ray_coordinate_ref = []
    near_ref, far_ref = pose_ref['near_fars'][ref_idx, 0], pose_ref['near_fars'][ref_idx, 1]
    ray_coordinate_world, ray_dir_world, colors, depth_candidates = [],[],[],[]
    rays_os, rays_ds, cos_angles, rays_depths = [],[],[],[]

    for i in range(V-1,V):
        intrinsic = intrinsics[i]  #!!!!!! assuming batch size equal to 1
        c2w, w2c = c2ws[i].clone(), w2cs[i].clone()

        rays_o, rays_d, pixel_coordinates = get_rays_mvs(H, W, intrinsic, c2w, N_rays, is_precrop_iters=is_precrop_iters)   # [N_rays 3]


        # direction
        ray_dir_world.append(rays_d)    # toward camera [N_rays 3]

        # position
        rays_o = rays_o.reshape(1, 3)
        rays_o = rays_o.expand(N_rays, -1)
        rays_os.append(rays_o)

        # colors
        pixel_coordinates_int = pixel_coordinates.long()
        color = imgs[0, i, :, pixel_coordinates_int[0], pixel_coordinates_int[1]] # [3 N_rays]
        colors.append(color)

        if depths.shape[2] != 1:
            rays_depth = depths[0,i,pixel_coordinates_int[0], pixel_coordinates_int[1]]
            rays_depths.append(rays_depth)

        # travel along the rays
        if with_depth:
            depth_candidate = near_fars[pixel_coordinates_int[0], pixel_coordinates_int[1]].reshape(-1,1) #  [ray_samples N_samples]
        else:
            if importanceSampling:
                near, far = rays_depth - 0.1, rays_depth + 0.1
                near, far = near.view(N_rays, 1), far.view(N_rays, 1)
            else:
                near, far = near_fars[0, i, 0], near_fars[0, i, 1]

            t_vals = torch.linspace(0., 1., steps=N_samples).view(1,N_samples).to(device)
            depth_candidate = near * (1. - t_vals) + far * (t_vals)
            depth_candidate = depth_candidate.expand([N_rays, N_samples])

            # get intervals between samples
            mids = .5 * (depth_candidate[..., 1:] + depth_candidate[..., :-1])
            upper = torch.cat([mids, depth_candidate[..., -1:]], -1)
            lower = torch.cat([depth_candidate[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(depth_candidate.shape, device=device)
            depth_candidate = lower + (upper - lower) * t_rand

        point_samples = rays_o.unsqueeze(1) + depth_candidate.unsqueeze(-1) * rays_d.unsqueeze(1)   #  [ray_samples N_samples 3 ]
        depth_candidates.append(depth_candidate) #  [ray_samples N_rays]

        # position
        ray_coordinate_world.append(point_samples)  # [ray_samples N_samples 3] xyz in [0,1]
        points_ndc = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=near_ref, far=far_ref, pad=pad)

        ray_coordinate_ref.append(points_ndc)

    ndc_parameters = {'w2c_ref':w2c_ref, 'intrinsic_ref':intrinsic_ref, 'inv_scale':inv_scale, 'near':near_ref, 'far':far_ref}
    colors = torch.cat(colors, dim=1).permute(1,0)
    rays_depths = torch.cat(rays_depths) if len(rays_depths)>0 else None
    depth_candidates = torch.cat(depth_candidates, dim=0)
    ray_dir_world = torch.cat(ray_dir_world, dim=0)
    ray_coordinate_world = torch.cat(ray_coordinate_world, dim=0)
    rays_os = torch.cat(rays_os, dim=0).permute(1,0)
    ray_coordinate_ref = torch.cat(ray_coordinate_ref, dim=0)

    return ray_coordinate_world, ray_dir_world, colors, ray_coordinate_ref, depth_candidates, rays_os, rays_depths, ndc_parameters

def build_rays_test(H,W, tgt_to_world, world_to_ref, intrinsic, near_fars_ref, near_fars, N_samples, pad=0, ref_idx=0, use_cpu=False, chunk=-1, idx=-1):
    '''

    Args:
        extrinsic intrinsic [4 4] [3 3)]
        N_samples: same as D int
        depth_values: [B D]

    Returns:
        [3 N_rays N_samples]
    '''

    device = torch.device("cpu") if use_cpu else tgt_to_world.device
    if use_cpu:
        tgt_to_world,world_to_ref = tgt_to_world.clone().to(device),world_to_ref.clone().to(device)
        intrinsic, near_fars_ref, near_fars = intrinsic.clone().to(device),near_fars_ref.clone().to(device),near_fars.clone().to(device)
    inv_scale = torch.tensor([W - 1, H - 1]).to(device)


    ray_coordinate_world, ray_dir_world, colors, depth_candidates = [],[],[],[]
    rays_os, rays_ds = [],[]

    intrinsic_render =  intrinsic if intrinsic.dim()==2 else intrinsic.mean(0)
    rays_o, rays_d, pixel_coordinates = get_rays_mvs(H, W, intrinsic_render, tgt_to_world, isRandom=False, chunk=chunk, idx=idx)
    ray_samples = H * W if chunk < 0 else pixel_coordinates.shape[-1]


    # direction
    ray_dir_world.append(rays_d)    # toward camera [N_rays 3]

    # position
    rays_o = rays_o.reshape(1,3)
    rays_o = rays_o.expand(ray_samples, -1)
    rays_os.append(rays_o)

    # travel along the rays
    near, far = near_fars[0], near_fars[1]
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    depth_candidate = near * (1. - t_vals) + far * (t_vals)
    depth_candidate = depth_candidate.expand([ray_samples, N_samples])
    point_samples = rays_o.unsqueeze(1) + depth_candidate.unsqueeze(-1) * rays_d.unsqueeze(1)    #  [3 ray_samples N_samples]
    depth_candidates.append(depth_candidate) #  [ray_samples N_rays]

    # position
    near, far = near_fars_ref[ref_idx, 0], near_fars_ref[ref_idx, 1]
    ray_coordinate_world.append(point_samples)  # [ray_samples N_samples 3] xyz in [0,1]
    ray_coordinate_ref = get_ndc_coordinate(world_to_ref, intrinsic, point_samples, inv_scale, near=near, far=far, pad=pad)

    ndc_parameters = {'w2c_ref': world_to_ref, 'intrinsic_ref': intrinsic, 'inv_scale': inv_scale, 'near': near, 'far': far}
    depth_candidates = torch.cat(depth_candidates, dim=0)
    ray_dir_world = torch.cat(ray_dir_world, dim=0)
    ray_coordinate_world = torch.cat(ray_coordinate_world, dim=0)
    rays_os = torch.cat(rays_os, dim=0)

    return ray_coordinate_world, ray_dir_world,ray_coordinate_ref, depth_candidates, rays_os, ndc_parameters


def build_color_volume(point_samples, pose_ref, imgs, img_feat=None, downscale=1.0, with_mask=False):
    '''
    point_world: [N_ray N_sample 3]
    imgs: [N V 3 H W]
    '''

    device = imgs.device
    N, V, C, H, W = imgs.shape
    inv_scale = torch.tensor([W - 1, H - 1]).to(device)

    C += with_mask
    C += 0 if img_feat is None else img_feat.shape[2]
    colors = torch.empty((*point_samples.shape[:2], V*C), device=imgs.device, dtype=torch.float)
    for i,idx in enumerate(range(V)):

        w2c_ref, intrinsic_ref = pose_ref['w2cs'][idx], pose_ref['intrinsics'][idx].clone()  # assume camera 0 is reference
        point_samples_pixel = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale)[None]
        grid = point_samples_pixel[...,:2]*2.0-1.0

        # img = F.interpolate(imgs[:, idx], scale_factor=downscale, align_corners=True, mode='bilinear',recompute_scale_factor=True) if downscale != 1.0 else imgs[:, idx]
        data = F.grid_sample(imgs[:, idx], grid, align_corners=True, mode='bilinear', padding_mode='border')
        if img_feat is not None:
            data = torch.cat((data,F.grid_sample(img_feat[:,idx], grid, align_corners=True, mode='bilinear', padding_mode='zeros')),dim=1)

        if with_mask:
            in_mask = ((grid >-1.0)*(grid < 1.0))
            in_mask = (in_mask[...,0]*in_mask[...,1]).float()
            data = torch.cat((data,in_mask.unsqueeze(1)), dim=1)

        colors[...,i*C:i*C+C] = data[0].permute(1, 2, 0)
        del grid, point_samples_pixel, data

    return colors


def normal_vect(vect, dim=-1):
    return vect / (torch.sqrt(torch.sum(vect**2,dim=dim,keepdim=True))+1e-7)

def get_ptsvolume(H, W, D, pad, near_far, intrinsic, c2w):
    device = intrinsic.device
    near,far = near_far

    corners = torch.tensor([[-pad,-pad,1.0],[W+pad,-pad,1.0],[-pad,H+pad,1.0],[W+pad,H+pad,1.0]]).float().to(intrinsic.device)
    corners = torch.matmul(corners, torch.inverse(intrinsic).t())

    linspace_x = torch.linspace(corners[0, 0], corners[1, 0], W+2*pad)
    linspace_y = torch.linspace(corners[ 0, 1], corners[2, 1], H+2*pad)
    ys, xs = torch.meshgrid(linspace_y, linspace_x)  # HW
    near_plane = torch.stack((xs,ys,torch.ones_like(xs)),dim=-1).to(device)*near
    far_plane = torch.stack((xs,ys,torch.ones_like(xs)),dim=-1).to(device)*far

    linspace_z = torch.linspace(1.0, 0.0, D).view(D,1,1,1).to(device)
    pts = linspace_z*near_plane + (1.0-linspace_z)*far_plane
    pts = torch.matmul(pts.view(-1,3), c2w[:3,:3].t()) + c2w[:3,3].view(1,3)

    return pts.view(D*(H+pad*2),W+pad*2,3)

def index_point_feature(volume_feature, ray_coordinate_ref, chunk=-1):
        ''''
        Args:
            volume_color_feature: [B, G, D, h, w]
            volume_density_feature: [B C D H W]
            ray_dir_world:[3 ray_samples N_samples]
            ray_coordinate_ref:  [3 N_rays N_samples]
            ray_dir_ref:  [3 N_rays]
            depth_candidates: [N_rays, N_samples]
        Returns:
            [N_rays, N_samples]
        '''

        device = volume_feature.device
        H, W = ray_coordinate_ref.shape[-3:-1]


        if chunk != -1:
            features = torch.zeros((volume_feature.shape[1],H,W), device=volume_feature.device, dtype=torch.float, requires_grad=volume_feature.requires_grad)
            grid = ray_coordinate_ref.view(1, 1, 1, H * W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
            for i in range(0, H*W, chunk):
                features[:,i:i + chunk] = F.grid_sample(volume_feature, grid[:,:,:,i:i + chunk], align_corners=True, mode='bilinear')[0]
            features = features.permute(1,2,0)
        else:
            grid = ray_coordinate_ref.view(-1, 1, H,  W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
            features = F.grid_sample(volume_feature, grid, align_corners=True, mode='bilinear')[:,:,0].permute(2,3,0,1).squeeze()#, padding_mode="border"
        return features





def to_tensor_cuda(data, device, filter):
    for item in data.keys():

        if item in filter:
            continue

        if type(data[item]) is np.ndarray:
            data[item] = torch.tensor(data[item], dtype=torch.float32, device= device)
        else:
            data[item] = data[item].float().to(device)
    return data


def to_cuda(data, device, filter):
    for item in data.keys():
        if item in filter:
            continue

        data[item] = data[item].float().to(device)
    return data

def tensor_unsqueeze(data, filter):
    for item in data.keys():
        if item in filter:
            continue

        data[item] = data[item][None]
    return data

def filter_keys(dict):
    dict.pop('N_samples')
    if 'ndc' in dict.keys():
        dict.pop('ndc')
    if 'lindisp' in dict.keys():
        dict.pop('lindisp')
    return dict

def sub_selete_data(data_batch, device, idx, filtKey=[], filtIndex=['view_ids_all','c2ws_all','scan','bbox','w2ref','ref2w','light_id','ckpt','idx']):
    data_sub_selete = {}
    for item in data_batch.keys():
        data_sub_selete[item] = data_batch[item][:,idx].float() if (item not in filtIndex and torch.is_tensor(item) and item.dim()>2) else data_batch[item].float()
        if not data_sub_selete[item].is_cuda:
            data_sub_selete[item] = data_sub_selete[item].to(device)
    return data_sub_selete

def detach_data(dictionary):
    dictionary_new = {}
    for key in dictionary.keys():
        dictionary_new[key] = dictionary[key].detach().clone()
    return dictionary_new

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def gen_render_path_spherical(theta, phi, radius=1.0):
    blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w





from scipy.interpolate import CubicSpline
def gen_render_path_pixelNeRF(c2ws, N_views=30):
    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = np.array(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = N_views // 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([450.0, 450.0, 450.0, 450.0, 450.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat, bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        R = R.from_quat(new_q)
        t = R[:, 2] * scale
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t
        new_pose = c2ws[0,0] @ new_pose
        render_poses.append(new_pose)
    render_poses = torch.stack(render_poses, dim=0)
    return render_poses



#################################################  MVS  helper functions   #####################################
from kornia.utils import create_meshgrid

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory



        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid


###############################  render path  ####################################
def pose_spherical_nerf(euler, radius=4.0):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,-radius])
    return c2ws_render

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def pose_spherical_dtu(radii, focus_depth, n_poses=120, world_center=np.array([0,0,0])):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center+world_center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0) @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])  # (n_poses, 3, 4)

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from warmup_scheduler import GradualWarmupScheduler
def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)

    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)
    return scheduler


####  pairing  ####
def get_nearest_pose_ids(tar_pose, ref_poses, num_select):
    '''
    Args:
        tar_pose: target pose [N, 4, 4]
        ref_poses: reference poses [M, 4, 4]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''

    dists = np.linalg.norm(tar_pose[:,None,:3,3] - ref_poses[None,:,:3,3], axis=-1)

    sorted_ids = np.argsort(dists, axis=-1)
    selected_ids = sorted_ids[:,:num_select]
    return selected_ids






def rotate_unitsphere(pose_ref, point_samples, ref_idx=0):
    '''
        point_samples [N_rays N_sample 3]
    '''

    c2w_ref = pose_ref['c2ws'][ref_idx]

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    camera_pos = c2w_ref[:3,3]
    ay = math.atan2(camera_pos[0],camera_pos[2])
    camera_pos_ry = rotate_2d(1, camera_pos, ay)
    ax = math.atan2(camera_pos_ry[1],camera_pos_ry[2])
    # camera_pos_ry_rx = rotate_2d(0, camera_pos_ry, ax)

    point_samples = rotate_2d(1, point_samples, ay)
    point_samples = rotate_2d(0, point_samples, ax)
    
    point_samples = point_samples.view(N_rays, N_samples, 3)

    return point_samples

def inverse_rotate_unitsphere(pose_ref, point_samples, ref_idx=0):
    '''
        point_samples [N_rays N_sample 3]
    '''

    c2w_ref = pose_ref['c2ws'][ref_idx]

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    camera_pos = c2w_ref[:3,3]
    ay = math.atan2(camera_pos[0],camera_pos[2])
    camera_pos_ry = rotate_2d(1, camera_pos, ay)
    ax = math.atan2(camera_pos_ry[1],camera_pos_ry[2])
    # camera_pos_ry_rx = rotate_2d(0, camera_pos_ry, ax)

    point_samples = rotate_2d(0, point_samples, -ax)
    point_samples = rotate_2d(1, point_samples, -ay)
    
    point_samples = point_samples.view(N_rays, N_samples, 3)

    return point_samples

def rotate_unitsphere_2D(pose_ref, point_samples, ref_idx=0):
    '''
        point_samples [N_rays N_sample 3]
    '''

    c2w_ref = pose_ref['c2ws'][ref_idx]

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    camera_pos = c2w_ref[:3,3]
    ay = math.atan2(camera_pos[0],camera_pos[2])

    point_samples = rotate_2d(1, point_samples, ay)
    
    point_samples = point_samples.view(N_rays, N_samples, 3)

    return point_samples

def inverse_rotate_unitsphere_2D(pose_ref, point_samples, ref_idx=0):
    '''
        point_samples [N_rays N_sample 3]
    '''

    c2w_ref = pose_ref['c2ws'][ref_idx]

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    camera_pos = c2w_ref[:3,3]
    ay = math.atan2(camera_pos[0],camera_pos[2])

    point_samples = rotate_2d(1, point_samples, -ay)
    
    point_samples = point_samples.view(N_rays, N_samples, 3)

    return point_samples

def rotate_2d(axis, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)

    R = torch.eye(3).cuda()
    if axis == 0:
        R[1,1],R[1,2],R[2,1],R[2,2] = cosa,sina,-sina,cosa
    elif axis == 1:
        R[0,0],R[0,2],R[2,0],R[2,2] = cosa,sina,-sina,cosa
    else:
        R[0,0],R[0,1],R[1,0],R[1,1] = cosa,sina,-sina,cosa

    points = points @ R

    return points

def world_to_sdf_input_space(pose_ref, point_samples, args, inv_scale=None):
    ref_idx = -1
    pad = 24
    # points = points_w2c_unitspshere(pose_ref, point_samples, inv_scale, ref_idx)
    # points = get_ndc_coordinate(pose_ref['w2cs'][ref_idx], pose_ref['intrinsics'][ref_idx], point_samples, inv_scale, near=pose_ref['near_fars'][ref_idx][0], far=pose_ref['near_fars'][ref_idx][1], pad=pad)
    # points = rotate_unitsphere(pose_ref, point_samples, ref_idx)
    if args.rotate_space:
        points = rotate_unitsphere_2D(pose_ref, point_samples, ref_idx)
    else:
        points = point_samples
    return points

def world_to_sdf_input_space_dirs(pose_ref, point_samples, args, inv_scale=None): 
    zero_points = torch.zeros_like(point_samples)
    dir_points = world_to_sdf_input_space(pose_ref, point_samples, args, inv_scale)
    zero_points = world_to_sdf_input_space(pose_ref, zero_points, args, inv_scale)
    dirs = dir_points - zero_points

    # dirs = dirs / (torch.sum(dirs, dim=-1, keepdim=True) + 1e-5)

    # ref_idx = 0
    # nr,ns,_ = point_samples.shape
    # dirs = point_samples.reshape(-1, 3) @ pose_ref['w2cs'][ref_idx][:3,:3].t()
    # dirs = dirs.reshape(nr, ns, 3)
    return dirs

def sdf_input_space_to_world(pose_ref, point_samples, args, inv_scale=None):
    ref_idx = -1
    pad = 24
    # points = points_c2w_unitspshere(pose_ref, point_samples, inv_scale, ref_idx)
    # points = inverse_get_ndc_coordinate(pose_ref['c2ws'][ref_idx], pose_ref['intrinsics'][ref_idx], point_samples, inv_scale, near=pose_ref['near_fars'][ref_idx][0], far=pose_ref['near_fars'][ref_idx][1], pad=pad)
    # points = inverse_rotate_unitsphere(pose_ref, point_samples, ref_idx)
    if args.rotate_space:
        points = inverse_rotate_unitsphere_2D(pose_ref, point_samples, ref_idx)
    else:
        points = point_samples
    return points

def sdf_input_space_to_world_dirs(pose_ref, point_samples, args, inv_scale=None):
    zero_points = torch.zeros_like(point_samples)
    dir_points = sdf_input_space_to_world(pose_ref, point_samples, args, inv_scale)
    zero_points = sdf_input_space_to_world(pose_ref, zero_points, args, inv_scale)
    dirs = dir_points - zero_points

    # dirs = dirs / (torch.sum(dirs, dim=-1, keepdim=True) + 1e-5)

    # ref_idx = 0
    # nr,ns,_ = point_samples.shape
    # dirs = point_samples.reshape(-1, 3) @ pose_ref['c2ws'][ref_idx][:3,:3].t()
    # dirs = dirs.reshape(nr, ns, 3)
    return dirs

def print_frustum(pose_ref, inv_scale, ref_idx=0):

    intrinsic_ref = pose_ref['intrinsics'][ref_idx]
    near, far = pose_ref['near_fars'][ref_idx]
    w2c_ref = pose_ref['w2cs'][ref_idx]
    c2w_ref = pose_ref['c2ws'][ref_idx]

    lindisp = False
    pad = 0

    frustum_corners_ndc = torch.tensor([[[-1,-1,0],[-1,-1,1],[1,1,0],[1,1,1],[0,0,0],[0,0,0.5],[0,0,1]]]).cuda()
    frustum_corners_world = inverse_get_ndc_coordinate(c2w_ref, intrinsic_ref, frustum_corners_ndc, inv_scale, near, far)
    frustum_corners_world = frustum_corners_world.squeeze()
    scale = torch.sum((frustum_corners_world[5,:] - frustum_corners_world[1,:])**2)
    frustum_middle_world = frustum_corners_world[5,:]
    direction_world = frustum_corners_world[6,:] - frustum_corners_world[5,:]
    direction_world = direction_world / direction_world.abs().sum()
    camera_pos_world = c2w_ref[:3,3]

    print(frustum_corners_world)
    return

def gen_pts_feats_no_ndc(imgs, volume_feature, rays_pts, pose_ref, feat_dim, img_feat=None, img_downscale=1.0, use_color_volume=False, net_type='v0', ref_idx=0, pad=0):

    rays_ndc = get_ndc_points(pose_ref, rays_pts, imgs, ref_idx=ref_idx, pad=pad)

    N_rays, N_samples = rays_pts.shape[:2]
    if img_feat is not None:
        feat_dim += img_feat.shape[1]*img_feat.shape[2]

    if not use_color_volume:
        input_feat = torch.empty((N_rays, N_samples, feat_dim), device=imgs.device, dtype=torch.float)
        ray_feats = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
        input_feat[..., :8] = ray_feats
        input_feat[..., 8:] = build_color_volume(rays_pts, pose_ref, imgs, img_feat, with_mask=True, downscale=img_downscale)
    else:
        input_feat = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)

    return input_feat

def get_ndc_points(pose_ref, rays_pts, imgs, ref_idx=0, pad=0):
    N, V, C, H, W = imgs.shape

    w2c_ref, intrinsic_ref = pose_ref['w2cs'][ref_idx], pose_ref['intrinsics'][ref_idx]  # assume camera 0 is reference
    inv_scale = torch.tensor([W-1, H-1]).cuda()

    ray_coordinate_ref = []
    near_ref, far_ref = pose_ref['near_fars'][ref_idx, 0], pose_ref['near_fars'][ref_idx, 1]

    points_ndc = get_ndc_coordinate(w2c_ref, intrinsic_ref, rays_pts, inv_scale, near=near_ref, far=far_ref, pad=pad) #ndc = normalized device coordinates

    return points_ndc

def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far

def gen_pts_neus(rays_o, rays_d, imgs, volume_feature, pose_ref, near_far_target, args, perturb, sdf_network, network_query_fn, n_samples=64, n_outside=32, n_importance=32, up_sample_steps=4):
    batch_size, _ = rays_o.shape
    N, V, C, H, W = imgs.shape
    inv_scale = torch.tensor([W-1, H-1]).cuda()
    near, far = near_far_from_sphere(rays_o, rays_d)
    if args.frustum_sampling:
        near, far = near_far_target
        n_samples = 64
        n_importance = 64
        n_outside = 0
        up_sample_steps = 4
    #sample_dist = 2.0 / n_samples   # Assuming the region of interest is a unit sphere
    # near, far = ear_far_target
    # z_vals = torch.linspace(0.0, 1.0, n_samples).cuda()
    # z_vals = near.cuda() + (far.cuda() - near.cuda()) * z_vals[None, :].cuda()

    t_vals = torch.linspace(0., 1., steps=n_samples).view(1,n_samples).cuda()
    depth_candidate = near * (1. - t_vals) + far * (t_vals)
    depth_candidate = depth_candidate.expand([batch_size, n_samples])
    z_vals = depth_candidate

    z_vals_outside = None
    if n_outside > 0:
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_outside + 1.0), n_outside)

    if perturb > 0:
        # t_rand = (torch.rand([batch_size, 1]) - 0.5).cuda()
        # z_vals = z_vals + t_rand * 2.0 / n_samples

        mids = .5 * (depth_candidate[..., 1:] + depth_candidate[..., :-1])
        upper = torch.cat([mids, depth_candidate[..., -1:]], -1)
        lower = torch.cat([depth_candidate[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(depth_candidate.shape).cuda()
        z_vals = lower + (upper - lower) * t_rand

        if n_outside > 0:
            mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
            z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

    if n_outside > 0:
        z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]).cuda() + 1.0 / n_samples

    background_alpha = None
    background_sampled_color = None

    inv_s = sdf_network.nerf.nerf_fg.deviation_network.forward(torch.ones([1]).cuda())
    base_inv_s = inv_s // 2

    # Up sample
    if n_importance > 0:
        with torch.no_grad():
            pts_world = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            pts = world_to_sdf_input_space(pose_ref, pts_world, args, inv_scale)

            pts_norm = torch.linalg.norm(pts, ord=2, dim=-1)
            inside_sphere = pts_norm < 1.0

            input_feat = gen_pts_feats_no_ndc(imgs, volume_feature, pts_world,
                                                    pose_ref,
                                                    args.feat_dim,
                                                    img_feat=None,
                                                    img_downscale=args.img_downscale,
                                                    use_color_volume=args.use_color_volume,
                                                    net_type=args.net_type,
                                                    ref_idx=0,
                                                    pad=args.pad)

            # input views are not used in sdf -> pass anything as input views
            sdf = network_query_fn(pts, pts, input_feat, sdf_network.sdf).squeeze()
            #sdf[~inside_sphere] = 100

            for i in range(up_sample_steps):
                new_z_vals = up_sample(rays_o,
                                            rays_d,
                                            z_vals,
                                            sdf,
                                            n_importance // up_sample_steps,
                                            base_inv_s * 2**i,
                                            pts)

                new_pts_world = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
                new_pts = world_to_sdf_input_space(pose_ref, new_pts_world, args, inv_scale)

                input_feat = gen_pts_feats_no_ndc(imgs, volume_feature, new_pts_world,
                                                    pose_ref,
                                                    args.feat_dim,
                                                    img_feat=None,
                                                    img_downscale=args.img_downscale,
                                                    use_color_volume=args.use_color_volume,
                                                    net_type=args.net_type,
                                                    ref_idx=0,
                                                    pad=args.pad)

                z_vals, sdf = cat_z_vals(sdf_network,
                                         rays_o,
                                         rays_d,
                                         z_vals,
                                         new_z_vals,
                                         sdf,
                                         network_query_fn,
                                         input_feat,
                                         new_pts,
                                         last=(i + 1 == up_sample_steps))

                pts_world = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts = world_to_sdf_input_space(pose_ref, pts_world, args, inv_scale)
                pts_ndc = get_ndc_points(pose_ref, pts_world, imgs, ref_idx=0, pad=args.pad)

    # Background model
    if n_outside > 0:
        z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
        z_vals, _ = torch.sort(z_vals_feed, dim=-1)

    pts_world = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
    pts_ndc = get_ndc_points(pose_ref, pts_world, imgs, ref_idx=0, pad=args.pad)

    return pts_world.detach(), z_vals.detach(), pts_ndc.detach()

def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s, pts):
    """
    Up sampling give a fixed inv_s
    """
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
    inside_sphere = (radius[:, :-1] < 1.42) | (radius[:, 1:] < 1.42)
    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).cuda(), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples

def cat_z_vals(sdf_network, rays_o, rays_d, z_vals, new_z_vals, sdf, network_query_fn, input_feat, new_pts, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts_world = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    pts = new_pts

    if not last:
        # new_sdf = sdf_network.sdf(pts.reshape(-1, 3),input_feat.reshape(-1,input_feat.size()[-1])).reshape(batch_size, n_importance)
        new_sdf = network_query_fn(pts, pts, input_feat, sdf_network.sdf).squeeze()
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1).cuda()
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).cuda()
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1).cuda(), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds).cuda(), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom).cuda(), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

import trimesh
import mcubes

def validate_mesh(base_dir_path, sdf, iter_step, world_space=False, resolution=100, threshold=0.0):
    bound_min = torch.tensor([-1,-1,-1], dtype=torch.float32)
    bound_max = torch.tensor([1,1,1], dtype=torch.float32)

    vertices, triangles =\
        extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, sdf=sdf)
    os.makedirs(os.path.join(base_dir_path, 'meshes'), exist_ok=True)

    # if world_space:
    #     vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(os.path.join(base_dir_path, 'meshes', '{:0>8d}.ply'.format(iter_step)))

    print('End 3d mesh creation')

def extract_geometry(bound_min, bound_max, resolution, threshold, sdf):
    u = sdf
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def point_grid(bound_min, bound_max, resolution):

    X = torch.linspace(bound_min[0], bound_max[0], resolution)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution)

    gridpoints = torch.cartesian_prod(X,Y,Z)

    gridpoints = gridpoints.reshape([resolution*resolution,resolution,3])
    rays_o = torch.clone(gridpoints[:,0,:]).squeeze()
    rays_o[:,2] = 0
    rays_d = torch.clone(gridpoints[:,0,:]).squeeze()
    rays_d[:,0:2] = 0
    rays_d[:,2] = 1

    z_vals = torch.clone(gridpoints[:,:,2]).squeeze()


    return gridpoints, rays_o, rays_d, z_vals
