# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import torch
from itertools import product
from torch import nn
import loss

"""
SDF utils for mesh file
"""


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


def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]
    return x, y, z


class GT_SDF(nn.Module):
    """
    Compute SDF value given (x, y, z) via RegularGridInterpolator
    """

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


class SDF:
    """
    Loads SDF from file and transform
    """

    def __init__(self, data_cfg, loss_cfg, device):
        super(SDF, self).__init__()
        self.sdf_map = GT_SDF(
            gt_sdf_file=data_cfg.gt_sdf_file,
            sdf_transf_file=data_cfg.sdf_transf_file,
            device=device,
        )
        self.loss_cfg = loss_cfg

    def sdf_eval_and_loss(
        self,
        sample,
    ):
        pc = sample["pc"]
        z_vals = sample["z_vals"]
        depth_sample = sample["depth_sample"]

        sdf = self.sdf_map(pc, None)  # SDF train
        sdf = sdf.reshape(pc.shape[:-1])

        # compute bounds
        bounds = loss.bounds_pc(pc, z_vals, depth_sample)
        # compute losses
        sdf_loss_mat, free_space_ixs = loss.sdf_loss(
            sdf, bounds, self.loss_cfg.trunc_distance, loss_type=self.loss_cfg.loss_type
        )

        total_loss, total_loss_mat, losses = loss.tot_loss(
            sdf_loss_mat,
            free_space_ixs,
            self.loss_cfg.trunc_weight,
        )

        return (
            total_loss,
            total_loss_mat,
            losses,
        )
