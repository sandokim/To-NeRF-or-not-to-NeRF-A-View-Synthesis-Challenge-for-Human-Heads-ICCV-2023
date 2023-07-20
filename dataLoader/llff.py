import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import json

from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


class LLFFDataset(Dataset):
    def __init__(self, datadir, split='train', is_toyEx=False, downsample=4, is_stack=False, hold_every=100, iteration=0, n_iters=0, view_exclude=False, view_exclude_list=[], nearfar_annealing=False, n_steps=2000):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        
        self.is_toyEx = is_toyEx
        
        self.iteration = iteration
        self.n_iters = n_iters
        self.n_steps = n_steps
        
        self.nearfar_annealing = nearfar_annealing
        
        self.view_exclude = view_exclude
        self.view_exclude_list = view_exclude_list

        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        #         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        # self.near_far = [0.0, 1.0] # default
        self.near_far = [3.5, 7.0] # ICCV2023
        
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        # self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    @staticmethod
    def anneal_nearfar(it, near_final, far_final, # https://github.com/Jiawei-Yang/FreeNeRF
                    n_steps=2000, init_perc=0.2, mid_perc=0.5):
        """Anneals near and far plane."""
        mid = near_final + mid_perc * (far_final - near_final)

        near_init = mid + init_perc * (near_final - mid)
        far_init = mid + init_perc * (far_final - mid)

        weight = min(it * 1.0 / n_steps, 1.0)

        near_i = near_init + weight * (near_final - near_init)
        far_i = far_init + weight * (far_final - far_init)
        
        near_far = [near_i, far_i]
        
        return near_far
    
    def read_meta(self):
        
        if self.is_toyEx:
            if self.split == 'train':
                poses_bounds = np.load(os.path.join(self.root_dir,'poses_bounds_train.npy'))
                self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                if self.view_exclude:
                    with open(os.path.join(self.root_dir,'transforms_train.json'), 'r') as f:
                        transforms_data = json.load(f)
                    file_paths = [data['file_path'] for data in transforms_data['frames']]
                    frame_numbers = [str(file_path.split('_')[-1]) for file_path in file_paths]
                    print("view_exclude_list", self.view_exclude_list)
                    print("frame_numbers", frame_numbers)
                    indices_to_exclude = [frame_numbers.index(idx) for idx in self.view_exclude_list if idx in frame_numbers]
                    print("exclude indices in poses_bounds", indices_to_exclude)
                    poses_bounds = np.delete(poses_bounds, indices_to_exclude, axis=0) 
                    # Exclude image paths
                    self.image_paths = [path for path in self.image_paths if path.split('.jpg')[0][-2:] not in self.view_exclude_list]
                    img_list = [path.split('.jpg')[0][-2:] for path in self.image_paths]
                    print("final_frame_numbers", img_list)
            else:
                poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds_train.npy'))  # (N_images, 17)
                self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                with open(os.path.join(self.root_dir,'transforms_train.json'), 'r') as f:
                    transforms_data = json.load(f)
                file_paths = [data['file_path'] for data in transforms_data['frames']]
                frame_numbers = [str(file_path.split('_')[-1]) for file_path in file_paths]
                val_views = ["04","12"]
                self.view_exclude_list = [num for num in frame_numbers if num not in val_views]
                print("val_views", val_views)
                print("view_exclude_list", self.view_exclude_list)
                print("frame_numbers", frame_numbers)
                indices_to_exclude = [frame_numbers.index(idx) for idx in self.view_exclude_list if idx in frame_numbers]
                print("exclude indices in poses_bounds", indices_to_exclude)
                poses_bounds = np.delete(poses_bounds, indices_to_exclude, axis=0) 
                # Exclude image paths
                self.image_paths = [path for path in self.image_paths if path.split('.jpg')[0][-2:] not in self.view_exclude_list]
                img_list = [path.split('.jpg')[0][-2:] for path in self.image_paths]
                print("final_frame_numbers", img_list)
        else:                
            if self.split == 'train':
                poses_bounds = np.load(os.path.join(self.root_dir,'poses_bounds_train.npy'))
                self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                if self.view_exclude:
                    with open(os.path.join(self.root_dir,'transforms_train.json'), 'r') as f:
                        transforms_data = json.load(f)
                    file_paths = [data['file_path'] for data in transforms_data['frames']]
                    frame_numbers = [str(file_path.split('_')[-1]) for file_path in file_paths]
                    print("view_exclude_list", self.view_exclude_list)
                    print("frame_numbers", frame_numbers)
                    indices_to_exclude = [frame_numbers.index(idx) for idx in self.view_exclude_list if idx in frame_numbers]
                    print("exclude indices in poses_bounds", indices_to_exclude)
                    poses_bounds = np.delete(poses_bounds, indices_to_exclude, axis=0) 
                    # Exclude image paths
                    self.image_paths = [path for path in self.image_paths if path.split('.jpg')[0][-2:] not in self.view_exclude_list]
                    img_list = [path.split('.jpg')[0][-2:] for path in self.image_paths]
                    print("final_frame_numbers", img_list)
            elif self.split == 'val':
                poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds_test.npy'))  # (N_images, 17)
                self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
            elif self.split == 'test':
                poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds_test.npy'))  # (N_images, 17)
                self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        
        # view check
        print('poses_bounds', len(poses_bounds))
        print('image_paths', len(self.image_paths))
            
        # load full resolution image then resize
        
        # Default Verifying a number of train images
        # if self.split in ['train', 'test']:
        #     assert len(poses_bounds) == len(self.image_paths), \
        #         'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        
        # default
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        
        # ICCV2023
        self.near_fars[:, -2:] = [0.4, 2.8]
        
        # near far annealing
        if self.nearfar_annealing:
            nf = LLFFDataset.anneal_nearfar(it=self.iteration, near_final=0.4, far_final=2.8, n_steps=self.n_steps)
            print('annealing near far', nf)
            self.near_fars[:, -2:] = nf
        
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        # self.poses, self.pose_avg = center_poses(poses, self.blender2opencv) # default
        self.poses = poses
        
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        for i in img_list:
            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            
            # default
            # rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)

            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample