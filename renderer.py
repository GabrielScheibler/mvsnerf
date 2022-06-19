from cmath import e

from numpy import ones_like
import torch
import torch.nn.functional as F
from utils import normal_vect, index_point_feature, build_color_volume, world_to_sdf_input_space, world_to_sdf_input_space_dirs

def depth2dist(z_vals, cos_angle):
    # z_vals: [N_ray N_sample]
    device = z_vals.device
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists

def ndc2dist(ndc_pts, cos_angle):
    dists = torch.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], dim=-1)
    dists = torch.cat([dists, 1e10*cos_angle.unsqueeze(-1)], -1)  # [N_rays, N_samples]
    return dists

def raw2alpha(sigma, dist, net_type):

    alpha_softmax = F.softmax(sigma, 1)

    alpha = 1. - torch.exp(-sigma) 

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T  # [N_rays, N_samples]
    return alpha, weights, alpha_softmax

def raw2alpha_neus(sigma_fg, sigma_bg, dist, net_type, inside_sphere):

    sigma_fg =  sigma_fg * inside_sphere
    sigma_bg = sigma_bg * (1.0 - inside_sphere)
    sigma = sigma_fg + sigma_bg

    alpha_softmax = F.softmax(sigma, 1)

    alpha = 1. - torch.exp(-sigma)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]

    weights = alpha * T  # [N_rays, N_samples]
    weights_fg = weights * inside_sphere
    weights_bg = weights * (1.0 - inside_sphere)

    return alpha, weights, alpha_softmax, weights_fg, weights_bg

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, alpha_only):
        if alpha_only:
            return torch.cat([fn.forward_alpha(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        else:
            return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret

def run_network_mvs(pts, viewdirs, alpha_feat, fn, embed_fn, embeddirs_fn, netchunk=1024):
    """Prepares inputs and applies network 'fn'.
    """

    if embed_fn is not None:
        pts = embed_fn(pts)

    if alpha_feat is not None:
        pts = torch.cat((pts,alpha_feat), dim=-1)

    if viewdirs is not None:
        if viewdirs.dim()!=3:
            viewdirs = viewdirs[:, None].expand(-1,pts.shape[1],-1)

        if embeddirs_fn is not None:
            viewdirs = embeddirs_fn(viewdirs)
        pts = torch.cat([pts, viewdirs], -1)

    alpha_only = viewdirs is None
    outputs_flat = batchify(fn, netchunk)(pts, alpha_only)
    outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def raw2outputs(raw, z_vals, dists, white_bkgd=False, net_type='v2'):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    device = z_vals.device

    rgb = raw[..., :3] # [N_rays, N_samples, 3]

    alpha, weights, alpha_softmax = raw2alpha(raw[..., 3], dists, net_type)  # [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map, disp_map, acc_map, weights, depth_map, alpha

def raw2outputs_neus(raw_fg, raw_bg, z_vals, dists, inside_sphere, white_bkgd=False, net_type='v2'):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device = z_vals.device

    #raw_bg[..., :] = 0

    alpha, weights, alpha_softmax, weights_fg, weights_bg = raw2alpha_neus(raw_fg[..., 3], raw_bg[..., 3], dists, net_type, inside_sphere)

    rgb_fg = raw_fg[..., :3] # [N_rays, N_samples, 3]
    rgb_bg = raw_bg[..., :3] # [N_rays, N_samples, 3]

    rgb_map = torch.sum(weights_fg[..., None] * rgb_fg, -2) + torch.sum(weights_bg[..., None] * rgb_bg, -2) # [N_rays, 3]
    depth_map = torch.sum(weights_fg * z_vals, -1) + torch.sum(weights_bg * z_vals, -1)

    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / (torch.sum(weights_fg, -1)+torch.sum(weights_bg, -1)))
    acc_map = torch.sum(weights_fg, -1) + torch.sum(weights_bg, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map, disp_map, acc_map, weights, depth_map, alpha

def transform_raw_neus(raw, pts, dirs, cos_anneal_ratio):
    # Section length
        batch_size, n_samples, _ = raw.shape

        dists = pts[:,1:,:] - pts[:,:-1,:]
        dists = torch.linalg.norm(dists, ord=2, dim=-1)
        sample_dist = torch.mean(dists).cuda()
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1).unsqueeze(-1)

        sampled_color, sdf, gradients, inv_s = torch.split(raw, [3, 1, 3, 1], dim=-1)

        dirs = dirs.unsqueeze(1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0 + 1e-07, 1.0 - 1e-07)

        sigma = - torch.log(1-alpha).reshape(batch_size, n_samples, 1)

        return torch.cat([sampled_color, sigma], -1), gradients, torch.mean(inv_s)

def gen_angle_feature(c2ws, rays_pts, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]

    Returns:

    """
    N_rays, N_samples = rays_pts.shape[:2]
    dirs = normal_vect(rays_pts.unsqueeze(2) - c2ws[:3, :3, 3][None, None])  # [N_rays, N_samples, v, 3]
    angle = torch.sum(dirs[:, :, :3] * rays_dir.reshape(N_rays,1,1,3), dim=-1, keepdim=True).reshape(N_rays, N_samples, -1)
    return angle

def gen_dir_feature(w2c_ref, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]

    Returns:

    """
    dirs = rays_dir @ w2c_ref[:3,:3].t() # [N_rays, 3]
    return dirs

def gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, feat_dim, img_feat=None, img_downscale=1.0, use_color_volume=False, net_type='v0'):
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

def rendering(args, pose_ref, rays_pts, rays_pts_ndc, depth_candidates, rays_o, rays_dir, inv_scale, cos_anneal_ratio,
              volume_feature=None, imgs=None, network_fn=None, img_feat=None, network_query_fn=None, white_bkgd=False, **kwargs):

    batch_size, n_samples, _ = rays_pts.size()

    # rays angle
    cos_angle = torch.norm(rays_dir, dim=-1)


    # using direction
    if pose_ref is not None:
        angle = gen_dir_feature(pose_ref['w2cs'][0], rays_dir/cos_angle.unsqueeze(-1))  # view dir feature
    else:
        angle = rays_dir/cos_angle.unsqueeze(-1)

    # rays_pts
    input_feat = gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_pts_ndc, args.feat_dim, \
                               img_feat, args.img_downscale, args.use_color_volume, args.net_type)

    # rays_dir_n = rays_dir/cos_angle.unsqueeze(-1)
    # rays_dir_n = rays_dir_n[:, None, :].expand(-1,rays_pts.shape[1],-1)
    # angle = angle[:, None, :].expand(-1,rays_pts.shape[1],-1)
    # pts = world_to_sdf_input_space(pose_ref, rays_pts, inv_scale)
    # dirs = world_to_sdf_input_space_dirs(pose_ref, rays_dir_n, inv_scale)

    # rays_ndc = rays_ndc * 2 - 1.0
    if 'neus' in args.net_type :
        H, W = imgs.shape[-2:]
        H, W = int(H), int(W)
        inv_scale = torch.tensor([W-1, H-1]).cuda()
        
        input_pts = world_to_sdf_input_space(pose_ref, rays_pts, args, inv_scale)
        input_dir = world_to_sdf_input_space_dirs(pose_ref, rays_dir[:,None,:], args, inv_scale).squeeze(1)

        pts_norm = torch.linalg.norm(rays_pts, ord=2, dim=-1)
        inside_sphere = (pts_norm < 1.0).float().detach()

        raw_bg = network_query_fn(rays_pts_ndc, angle, input_feat, network_fn.forward_bg)
        raw_fg_alpha = network_query_fn(input_pts, input_dir, input_feat, network_fn.forward_fg)
        raw_fg, sdf_gradients, inv_s = transform_raw_neus(raw_fg_alpha, input_pts, input_dir, cos_anneal_ratio)

        sdf_gradient_error = (torch.linalg.norm(sdf_gradients, ord=2, dim=-1) - 1.0) ** 2
        sdf_gradient_error = sdf_gradient_error * inside_sphere
        sdf_gradient_error = sdf_gradient_error.sum() / inside_sphere.sum()

        dists = depth2dist(depth_candidates, cos_angle)
        # dists = ndc2dist(rays_ndc)
        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs_neus(raw_fg, raw_bg, depth_candidates, dists, inside_sphere, white_bkgd, args.net_type)
        rgb_fg, _, _, _, _, _ = raw2outputs_neus(raw_fg, torch.zeros_like(raw_bg), depth_candidates, dists, inside_sphere, white_bkgd, args.net_type)
        rgb_bg, _, _, _, _, _ = raw2outputs_neus(torch.zeros_like(raw_fg), raw_bg, depth_candidates, dists, inside_sphere, white_bkgd, args.net_type)
        ret = {'inv_s' : inv_s}
        return rgb_map, input_feat, weights, depth_map, alpha, ret, rgb_fg, rgb_bg, sdf_gradient_error


    raw = network_query_fn(rays_pts_ndc, angle, input_feat, network_fn)
    if raw.shape[-1]>4:
        input_feat = torch.cat((input_feat[...,:8],raw[...,4:]), dim=-1)

    dists = depth2dist(depth_candidates, cos_angle)
    # dists = ndc2dist(rays_ndc)
    rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, depth_candidates, dists, white_bkgd,args.net_type)
    ret = {}

    return rgb_map, input_feat, weights, depth_map, alpha, ret, rgb_map, rgb_map, torch.ones_like(rays_pts).cuda()

def mesh_rendering(args, pose_ref, rays_pts, rays_pts_ndc, inv_scale,
              volume_feature=None, imgs=None, network_fn=None, img_feat=None, network_query_fn=None, white_bkgd=False, **kwargs):

    # rays_pts
    input_feat = gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_pts_ndc, args.feat_dim, \
                            img_feat, args.img_downscale, args.use_color_volume, args.net_type)

    # rays_ndc = rays_ndc * 2 - 1.0

    H, W = imgs.shape[-2:]
    H, W = int(H), int(W)
    inv_scale = torch.tensor([W-1, H-1]).cuda()
    
    input_pts = world_to_sdf_input_space(pose_ref, rays_pts, args, inv_scale)
    input_dir = input_pts # dirs are not used for sdf and their values do not matter

    if 'neus' in args.net_type:
        sdf = network_query_fn(input_pts, input_dir, input_feat, network_fn.sdf)
        return sdf
    else:
        raw = network_query_fn(rays_pts_ndc, input_dir, input_feat, network_fn)
        alpha, _, _ = raw2alpha(raw[...,3], None, None)
        return alpha

def render_density(network_fn, rays_pts, density_feature,  network_query_fn, chunk=1024 * 5):
    densities = []
    device = density_feature.device
    for i in range(0, rays_pts.shape[0], chunk):

        input_feat = rays_pts[i:i + chunk].to(device)

        density = network_query_fn(input_feat, None, density_feature[i:i + chunk], network_fn)
        densities.append(density)

    return torch.cat(densities)