#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import numpy as np
import copy
from tqdm import tqdm
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
import cv2

# ================================================================================
# Trajectory Generation Utils 
# (Adapted from Triangle Splatting utils/render_utils.py as they are not present 
# in the provided original Gaussian Splatting utils)
# ================================================================================

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
    return p[..., :3, :4]

def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def transform_poses_pca(poses: np.ndarray):
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray, n_frames: int = 120, z_variation: float = 0., z_phase: float = 0.) -> np.ndarray:
    center = focus_point_fn(poses)
    offset = np.array([center[0], center[1], 0])
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                            (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    positions = positions[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def generate_path(viewpoint_cameras, n_frames=240):
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj = []
    base_cam = viewpoint_cameras[0]
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1])
        cam = copy.deepcopy(base_cam)
        # Ensure dimensions are even for video encoding
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        
        # Update camera extrinsic parameters
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        traj.append(cam)

    return traj

# ================================================================================
# Main Script
# ================================================================================

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Video creation script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_as", default="output_video", type=str)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--n_frames", default=240, type=int, help="Number of frames for the video trajectory")
    args = get_combined_args(parser)
    print("Creating video for " + args.model_path)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    # Initialize Gaussian Model
    gaussians = GaussianModel(dataset.sh_degree)

    # Load Scene
    scene = Scene(args=dataset,
                  gaussians=gaussians,
                  load_iteration=args.iteration,
                  shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Setup output directories
    traj_dir = os.path.join(args.model_path, 'traj')
    os.makedirs(traj_dir, exist_ok=True)

    render_path = os.path.join(traj_dir, "renders")
    os.makedirs(render_path, exist_ok=True)
    
    # Generate smooth camera trajectory based on training views
    print("Generating camera trajectory...")
    train_cameras = scene.getTrainCameras()
    if len(train_cameras) == 0:
        print("No training cameras found to generate trajectory.")
        exit()
        
    cam_traj = generate_path(train_cameras, n_frames=args.n_frames)
    
    # Render frames
    print(f"Rendering {len(cam_traj)} frames...")
    with torch.no_grad():
        for idx, view in enumerate(tqdm(cam_traj, desc="Rendering progress")):
            # Gaussian Splatting render returns a dictionary
            rendering = render(view, gaussians, pipe, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    # Compile rendered images into a video using OpenCV
    print("Encoding video...")
    image_folder = render_path
    output_video = os.path.join(args.model_path, args.save_as + '.mp4')

    # Get all image files sorted by name
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("No images found to create video.")
        exit()

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, args.fps, (width, height))

    # Write each image to the video
    for img_name in tqdm(images, desc="Video encoding progress"):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()

    print(f'Video saved as {output_video}')