#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Localization in a neural field with depth data

import torch

import theseus as th
from sdf_utils import GT_SDF
from vision_dataset import VisionDataset
import os
import hydra
import random
import numpy as np
from sampler import Sampler


@hydra.main(config_path="../configs/", config_name="neural_field")
def main(cfg):

    device = cfg.device

    # Set seeds
    torch.set_default_dtype(torch.double)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_cfg, camera_cfg, sample_cfg = (cfg.data, cfg.camera, cfg.sampling)

    # 1. Load the ground-truth SDF of the scene (power_drill)
    sdf_map = GT_SDF(
        gt_sdf_file=data_cfg.gt_sdf_file,
        sdf_transf_file=data_cfg.sdf_transf_file,
        device=device,
    )

    # 2. Render ground-truth depth and pose, and noisy initial pose
    scene_dataset = VisionDataset(
        scene_file=data_cfg.scene_file, cfg=camera_cfg, device=device
    )
    posed_depth = scene_dataset[0]  # extract rgb, d, transform
    _, (_, depth, T_gt, T) = zip(*posed_depth.items())

    (_, H, W) = depth.shape
    # Define optimizer and run GN for matching T --> T_gt using depth + SDF rendering
    T_gt = th.SE3(tensor=T_gt.clone()[:, :3, :].double(), name="T_gt")
    T = th.SE3(tensor=T[:, :3, :].double(), name="T")
    depth = th.Vector(tensor=depth.view(1, -1).double(), name="depth")

    sampler = Sampler(sample_cfg, camera_cfg, device)
    optim_vars = (T,)
    aux_vars = (depth,)
    objective = th.Objective(dtype=torch.float64)

    """
    Define SDF loss function
    """

    def sdf_loss(optim_vars, aux_vars):
        (pose_batch,) = optim_vars  # poses
        (depth_batch,) = aux_vars  # depth

        # Loss wrt 3D sample points
        sample_pts = sampler.sample_points(
            depth_batch.tensor.view(-1, H, W),
            pose_batch,
        )

        (total_loss, total_loss_mat, _, _, _,) = sampler.sdf_eval_and_loss(
            sample_pts,
            vision_weights=None,
            do_avg_loss=True,
        )

        return total_loss.view(1, 1)

        # # Loss wrt 2D rendered depth image
        # depth_batch = depth_batch.tensor.view(-1, H, W)
        # render_depth = self.render_depth(pose_batch, sensor, depth_batch)
        # loss = torch.nn.functional.l1_loss(depth_batch, render_depth, reduction="none")
        # return loss.mean().view(1, 1)

    """
    Define cost function 
    """
    sdf_loss_cf = th.AutoDiffCostFunction(
        optim_vars,
        sdf_loss,
        1,
        aux_vars=aux_vars,
        name="sdf_loss",
        autograd_mode="dense",
    )
    objective.add(sdf_loss_cf)
    objective.to(device)

    print(objective.error())

    # optimizer = th.GaussNewton(
    #     objective,
    #     max_iterations=15,
    #     step_size=1e-5,
    # )
    # theseus_optim = th.TheseusLayer(optimizer)
    # theseus_optim.to(device)

    # # Optimize
    # theseus_inputs = {
    #     "T": T,
    #     "depth": depth,
    # }
    # # Optimize over N iterations
    # _, info = theseus_optim.forward(
    #     theseus_inputs,
    #     optimizer_kwargs={
    #         "track_best_solution": True,
    #         "damping": 0.1,
    #     },
    # )

    # 3. Sample from image sensor.sample_points() in sensor.py
    # 4. sdf_eval_and_loss from trainer.py


if __name__ == "__main__":
    main()
