#!/usr/bin/env python3

"""
Sample script to localize a camera in a neural-field (SDF) using sampled depth losses and auto-diff cost function 

python neural_field.py optimizer=ADAM 
TODO: shows local-minima problem with large gradient updates in translation over rotation, leading to wrong solution 
()

python neural_field.py optimizer=GN 
TODO: show ill-conditioned error with the AutoDiffCostFunction (The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: 2).)
"""

import torch
import theseus as th
from sdf_utils import SDF
from vision_dataset import VisionDataset
import hydra
import random
import numpy as np
from sampler import Sampler
from visualizer import Visualizer
from pose import camera_transf
from theseus import LieGroupTensor
import os


@hydra.main(version_base=None, config_path="../configs/", config_name="neural_field")
def main(cfg):
    device = cfg.device
    optimizer_type = cfg.optimizer
    visualize = cfg.visualize

    # Set seeds
    torch.set_default_dtype(torch.double)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    data_cfg, camera_cfg, sample_cfg, loss_cfg = (
        cfg.data,
        cfg.camera,
        cfg.sampling,
        cfg.loss,
    )

    """
    1. Load SDF ground-truth
    """
    sdf = SDF(data_cfg, loss_cfg, device)

    """
    2. Generate vision dataset from scene mesh, camera pose, and pyrender
    """
    scene_dataset = VisionDataset(
        scene_file=data_cfg.scene_file, cfg=camera_cfg, device=device
    )
    posed_depth = scene_dataset[0]
    _, (_, depth, T_gt, T) = zip(*posed_depth.items())
    (_, H, W) = depth.shape

    """
    3. Define ground-truth and optimization variables
    """

    T_gt = th.SE3(tensor=T_gt.clone()[:, :3, :].double(), name="T_gt")
    T = th.SE3(tensor=T[:, :3, :].double(), name="T")
    depth = th.Vector(tensor=depth.view(1, -1).double(), name="depth")

    """
    4. Load the o3d visualizer
    """
    if visualize:
        vis = Visualizer(
            camera_cfg=camera_cfg, scene_file=data_cfg.scene_file, T_gt=T_gt, T=T
        )

    """
    5. define depth sampling class
    """
    sampler = Sampler(sample_cfg, camera_cfg, device)
    optim_vars = (T,)
    aux_vars = (depth,)

    def sdf_loss(optim_vars, aux_vars):
        """
        6. SDF loss function for ADAM/GN optimizer
        """
        (pose_batch,) = optim_vars  # poses
        (depth_batch,) = aux_vars  # depth

        # Randomly sample non-zero pixels of the depth-image, and compute the backprojected (x, y, z) pointcloud
        sample_pts = sampler.sample_points(
            depth_batch.tensor.view(-1, H, W),
            pose_batch,
        )

        # Compute the TSDF loss of the (x, y, z) pointcloud wrt the SDF
        (loss, _, _) = sdf.sdf_eval_and_loss(
            sample_pts,
        )

        # Visualize the (x, y, z) pointcloud in o3d
        if visualize:
            vis.update_pc(sample_pts["pc"].clone().view(-1, 3).detach().cpu().numpy())

        if optimizer_type == "ADAM":
            return loss  # Returns scalar for ADAM optimizer
        else:
            return loss.view(1, 1)  # Returns (1, 1) matrix for GN optimizer

        # # TODO: Alternate formulation, loss wrt 3D->2D rendered depth image
        # depth_batch = depth_batch.tensor.view(-1, H, W)
        # render_depth = self.render_depth(pose_batch, sensor, depth_batch)
        # loss = torch.nn.functional.l1_loss(depth_batch, render_depth, reduction="mean")
        # return loss.view(1, 1)

    if optimizer_type == "ADAM":
        """
        7. ADAM optimizer: first order gradient descent steps using TSDF loss on exponential map
        """
        cam_transf = camera_transf().to(device)

        N = 1000

        optimizer = torch.optim.Adam(
            [
                {"params": [cam_transf.w, cam_transf.theta], "lr": 1e-2},
                {"params": [cam_transf.t], "lr": 5e-4},
            ],
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[N // 2, 2 * N // 3], gamma=0.5
        )

        optimizer.zero_grad()

        # Optimize over N iterations
        for i in range(N):
            cam_matrix = cam_transf(T.to_matrix())
            # cam_matrix.retain_grad()
            loss = sdf_loss((cam_matrix,), (depth,))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # compute the between pose error wrt ground-truth
            T_opt = th.SE3(tensor=cam_matrix[:, :3, :])
            pose_err = T_gt.local(T_opt)
            pose_err = (pose_err**2).sum(dim=1).mean()
            print(f"ADAM iteration {i}, loss: {loss.item()}, pose error: {pose_err}")
            print(
                "---------------------------------------------------------------"
                "---------------------------"
            )
            if visualize:
                vis.update_cam(T_opt)  # Visualize the new pose in o3d
    else:
        """
        8. Gauss newton optimizer: second-order using TSDF loss and AutoDiffCostFunction
        """
        objective = th.Objective(dtype=torch.float64)
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

        optimizer = th.GaussNewton(
            objective,
            max_iterations=15,
            step_size=1e-4,
        )
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim.to(device)

        N = 10
        for i in range(N):
            theseus_inputs = {
                "T": T,
                "depth": depth,
            }
            _, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "damping": 0.1,
                },
            )
            theseus_inputs["T"] = info.best_solution["T"].to(device)
            theseus_inputs["T"] = th.SE3(tensor=theseus_inputs["T"], name="T")

            T_opt = theseus_inputs["T"].clone()
            pose_err = T_gt.local(T_opt)
            pose_err = (pose_err**2).sum(dim=1).mean()
            print(f"GN outer-loop {i}, pose error: {pose_err}")
            print(
                "---------------------------------------------------------------"
                "---------------------------"
            )
            if visualize:
                vis.update_cam(T_opt)  # Visualize the new pose in o3d


if __name__ == "__main__":
    main()
