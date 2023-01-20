# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import git
import torch


# quicklink to the root and folder directories
root = git.Repo(".", search_parent_directories=True).working_tree_dir
import pyrender
import trimesh
import open3d as o3d
import open3d.core as o3c
import cv2
from torchvision import transforms
import theseus as th


class BGRtoRGB(object):
    """bgr format to rgb"""

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class DepthTransform(object):
    """gel2cam transformation"""

    def __init__(self, cam_dist):
        self.cam_dist = cam_dist

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        depth += self.cam_dist
        depth[depth == self.cam_dist] = np.nan
        return depth.astype(np.float32)


class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale


class VisionDataset(torch.utils.data.Dataset):
    def __init__(self, scene_file, cfg, device):
        W = cfg.intrinsics.w
        H = cfg.intrinsics.h
        yfov = cfg.intrinsics.yfov
        cam_dist = cfg.cam_dist
        inv_depth_scale = 1.0 / cfg.depth_scale
        rgb_transform = transforms.Compose([BGRtoRGB()])
        depth_transform = transforms.Compose(
            [
                DepthScale(inv_depth_scale),
                DepthTransform(cam_dist),
            ]
        )

        self.realsense = vision_sim(scene_file, W=W, H=H, yfov=yfov)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.length = 0
        self.noisy = True
        self.device = device

        # Add noise to pose
        self.rotation_noise = cfg.noise.rotation
        self.translation_noise = cfg.noise.translation

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, depth, pose = self.realsense.render(noisy=self.noisy)
        mask = depth > 5e-2
        depth = depth * mask  # apply contact mask
        pose = torch.from_numpy(pose).to(self.device)
        sample = {"image": image, "depth": depth, "T_gt": pose, "T": pose}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        self.length += 1

        sample["image"] = sample["image"][None, ...]  # (1, H, W, C)
        sample["depth"] = sample["depth"][None, ...]  # (1, H, W)
        sample["T_gt"] = sample["T_gt"][None, ...]  # (1, 4, 4)

        sample["image"] = (
            torch.from_numpy(sample["image"]).float().to(self.device) / 255.0
        )
        sample["depth"] = torch.from_numpy(sample["depth"]).float().to(self.device)

        # Add noise to pose
        pose_noise = th.SE3.exp_map(
            torch.cat(
                [
                    self.translation_noise * (2.0 * torch.rand(1, 3) - 1),
                    self.rotation_noise * (2 * torch.rand(1, 3) - 1),
                ],
                dim=1,
            ).to(self.device)
        ).to_matrix()
        sample["T"] = sample["T_gt"] @ pose_noise
        return sample


class vision_sim:
    def __init__(self, scene_file, W=1280, H=720, yfov=60):
        # dense point cloud with vertice information
        fuze_trimesh = trimesh.load(scene_file)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.scene = pyrender.Scene()
        self.scene.add(mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(yfov), aspectRatio=1.0)

        self.obj_center = mesh.centroid

        # Add noise with sigma
        self.color_noise = 5  # pixels
        self.depth_noise = 5e-3  # meters

        # Redwood Indoor LivingRoom1 (Augmented ICL-NUIM)
        # http://redwood-data.org/indoor/
        data = o3d.data.RedwoodIndoorOffice1()
        noise_model_path = data.noise_model_path
        im_src_path = data.depth_paths[0]
        self.simulator = o3d.t.io.DepthNoiseSimulator(noise_model_path)

        # # Read clean depth image (uint16)
        self.test_depth_clean = o3d.t.io.read_image(im_src_path)

        # # Pick camera viewing direction: MATLAB view(3)
        T_wc = np.zeros((4, 4))  # transform from point coord to world coord
        T_wc[3, 3] = 1.0
        T_wc[0:3, 3] = np.array(
            [0.5 + self.obj_center[0], 0.0 + self.obj_center[1], 0.5]
        )  # translation
        s = np.sqrt(2) / 2
        T_wc[0:3, 0:3] = np.array(
            [[0.0, -s, s], [1.0, 0.0, 0.0], [0.0, s, s]]
        )  # rotation

        self.camera_pose = T_wc

        self.scene.add(camera, pose=T_wc)
        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=3.0,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        self.scene.add(light, pose=T_wc)
        self.r = pyrender.OffscreenRenderer(W, H)

    def render(self, noisy=True):
        color_clean, depth_clean = self.r.render(self.scene)

        if not noisy:
            self.simulator.enable_deterministic_debug_mode()

        color_noise = np.random.normal(0, self.color_noise, color_clean.shape)
        color = np.clip(color_clean + color_noise, 0, 255).astype(np.uint8)

        og_shape = depth_clean.shape
        depth_clean = cv2.resize(
            depth_clean, (640, 480), interpolation=cv2.INTER_NEAREST
        )
        depth = o3d.t.geometry.Image(o3c.Tensor(depth_clean))
        depth = self.simulator.simulate(depth, depth_scale=1.0)  # 0.00137 avg depth
        depth = np.array(depth).squeeze()
        depth = cv2.resize(
            depth, (og_shape[1], og_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        depth_clean = cv2.resize(
            depth_clean, (og_shape[1], og_shape[0]), interpolation=cv2.INTER_NEAREST
        )

        return color, depth, self.camera_pose

    # https://github.com/mmatl/pyrender/issues/14#issuecomment-485881479
    def pointcloud(self, depth, fov):
        height = depth.shape[0]
        width = depth.shape[1]

        fy = fx = 0.5 / np.tan(fov * 0.5)  # assume aspectRatio is one.

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        normalized_y = -normalized_y

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        # world_z = depth[y, x]
        world_z = -depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T
