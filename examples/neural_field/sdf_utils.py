# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import torch

import torch
from itertools import product
from torch import nn


def sdf_eval_and_loss(
    self,
    sample,
    vision_weights=None,
    do_avg_loss=True,
):
    pc = sample["pc"]
    z_vals = sample["z_vals"]
    indices_b = sample["indices_b"]
    indices_h = sample["indices_h"]
    indices_w = sample["indices_w"]
    dirs_C_sample = sample["dirs_C_sample"]
    depth_sample = sample["depth_sample"]
    T_WC_sample = sample["T_WC_sample"]
    norm_sample = sample["norm_sample"]
    binary_masks = sample["binary_masks"]
    depth_batch = sample["depth_batch"]
    format = sample["format"]

    loss_params = self.loss_params

    do_sdf_grad = loss_params.eik_weight != 0
    if do_sdf_grad:
        pc.requires_grad_()

    if "realsense" in format:
        # add noise to prevent overfitting to high-freq data from a noisy sensor
        noise = torch.randn(pc.shape, device=pc.device) * self.noise_std
        pc = pc + noise

    # sdf and sdf gradient
    sdf = self.sdf_map(pc, None)  # SDF train
    sdf = sdf.reshape(pc.shape[:-1])

    sdf_grad = None
    if do_sdf_grad:
        sdf_grad = model.gradient(pc, sdf)

    # compute bounds
    bounds, grad_vec = loss.bounds(
        loss_params.bounds_method,
        dirs_C_sample,
        depth_sample,
        T_WC_sample,
        z_vals,
        pc,
        loss_params.trunc_distance,
        norm_sample,
        do_grad=False,
    )

    # compute loss

    # equation (8)
    sdf_loss_mat, free_space_ixs = loss.sdf_loss(
        sdf, bounds, loss_params.trunc_distance, loss_type=loss_params.loss_type
    )

    #### added, test
    eik_loss_mat = None
    if loss_params.eik_weight != 0:
        eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)

    if vision_weights is not None:
        vision_weights = vision_weights.reshape(sdf.shape)
    total_loss, total_loss_mat, losses = loss.tot_loss(
        sdf_loss_mat,
        eik_loss_mat,
        free_space_ixs,
        bounds,
        loss_params.trunc_weight,
        loss_params.eik_weight,
        vision_weights=vision_weights,
    )
    ####

    # total_loss, total_loss_mat, losses = loss.tot_loss(
    #     sdf_loss_mat,
    #     free_space_ixs,
    #     bounds,
    #     loss_params.trunc_weight,
    # )

    loss_approx, frame_avg_loss = None, None

    W, H, loss_approx_factor = (
        self.sensor[format].W,
        self.sensor[format].H,
        self.sensor[format].loss_approx_factor,
    )

    if do_avg_loss:
        loss_approx, frame_avg_loss = loss.frame_avg(
            total_loss_mat,
            depth_batch,
            indices_b,
            indices_h,
            indices_w,
            W,
            H,
            loss_approx_factor,
            binary_masks,
        )

    return (
        total_loss,
        total_loss_mat,
        losses,
        loss_approx,
        frame_avg_loss,
    )


class RegularGridInterpolator:
    """
    https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py
    """

    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x.contiguous(), p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.0
            dist_right[dist_right < 0] = 0.0
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.0

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.0
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return numerator / denominator


class GT_SDF(nn.Module):
    def __init__(self, gt_sdf_file, sdf_transf_file, device):
        super().__init__()
        sdf_grid = np.load(gt_sdf_file)
        sdf_transform = np.loadtxt(sdf_transf_file)

        x, y, z = get_grid_pts(sdf_grid.shape, sdf_transform)
        x, y, z = (
            torch.tensor(x).to(device).float(),
            torch.tensor(y).to(device).float(),
            torch.tensor(z).to(device).float(),
        )
        sdf_grid = torch.tensor(sdf_grid).to(device).float()
        self.sdf_interp = RegularGridInterpolator((x, y, z), sdf_grid)

    def forward(self, x, noise_std=None):
        x = x.reshape(-1, 3).float()  # [B, 3]
        (xx, yy, zz) = torch.tensor_split(x, 3, dim=1)
        h = self.sdf_interp((xx, yy, zz))
        h = h.squeeze()
        if noise_std is not None:
            noise = torch.randn(h.shape, device=h.device) * noise_std
            h = h + noise
        return h


# Helper scripts to load sdf from file


def sdf_interpolator(sdf_grid, transform):
    x, y, z = get_grid_pts(sdf_grid.shape, transform)
    sdf_interp = scipy.interpolate.RegularGridInterpolator((x, y, z), sdf_grid)
    return sdf_interp


def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]
    return x, y, z
