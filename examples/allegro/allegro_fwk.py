# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import theseus as th
from theseus.embodied.robot.forward_kinematics import Robot
from theseus.embodied.robot.forward_kinematics import (
    get_forward_kinematics,
)
import logging
from helpers import viz_robot, images_to_video, load_robot
import pathlib
import os

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# TODO: test higher batch sizes, with batchsize = 1 cpu is faster than gpu

# Load the robot model and forward kinematics
URDF_PATH = os.path.join("allegro_hand_description_left_digit.urdf")
robot, fkin, all_links, tip_ids = load_robot(URDF_PATH, device)
# tip_ids = range(len(all_links))
# Given N noisy ee_poses and less-noisy encoder values, optimize for the fk_cost
# Define cost (distance between desired and current ee pose)
def ee_pose_err_fn(optim_vars, aux_vars):
    (ee_poses,) = optim_vars
    (ee_poses_target,) = aux_vars
    pose_err = ee_poses_target.local(ee_poses)
    return pose_err


def main(visualize=False):

    n_links = len(all_links)
    NUM_DOFS = 16

    init_pose = torch.tensor(
        [
            0.0627,
            1.2923,
            0.3383,
            0.1088,
            0.0724,
            1.1983,
            0.1551,
            0.1499,
            0.1343,
            1.1736,
            0.5355,
            0.2164,
            1.1202,
            1.1374,
            0.8535,
            -0.0852,
        ],
        device=device,
    )  # init grasp pose for allegro
    # Allegro convention is index ('link_3.0_tip') middle ('link_7.0_tip') ring ('link_11.0_tip') thumb ('link_15.0_tip') order

    # FK function is applied breadth-first, so swap the indices from the allegro convention
    joint_map = torch.tensor(
        [joint.id for joint in robot.joint_map.values() if joint.id < NUM_DOFS],
        device=device,
    )
    # Swap index and ring for theseus left-allegro
    init_pose[[0, 1, 2, 3]], init_pose[[8, 9, 10, 11]] = (
        init_pose[[8, 9, 10, 11]],
        init_pose[[0, 1, 2, 3]],
    )
    init_pose = init_pose[joint_map]

    # Intialize ground truth link poses
    encoder_vals = th.Vector(tensor=init_pose, name="encoder_vals", dtype=torch.float64)
    gt_poses = fkin(encoder_vals.tensor)
    gt_poses = torch.vstack(gt_poses).to(device)
    gt_poses = th.SE3(tensor=gt_poses, name="poses", dtype=torch.float32)
    est_poses = gt_poses.copy()

    # Generate noisy observed poses
    translation_noise, rotation_noise = 5e-1, 5e-1
    noise = th.SE3.exp_map(
        torch.cat(
            [
                translation_noise * (2.0 * torch.rand(n_links, 3, device=device) - 1),
                rotation_noise * (2 * torch.rand(n_links, 3, device=device) - 1),
            ],
            dim=1,
        )
    )
    noisy_poses = th.compose(gt_poses, noise)

    # Set up optimization
    ee_poses = th.SE3(tensor=noisy_poses[tip_ids], name="ee_poses")
    ee_poses_target = fkin(encoder_vals.tensor)
    ee_poses_target = th.SE3(
        tensor=torch.vstack([ee_poses_target[i] for i in tip_ids]),
        name="ee_poses_target",
    )

    optim_vars = (ee_poses,)
    aux_vars = (ee_poses_target,)

    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        ee_pose_err_fn,
        6,
        aux_vars=aux_vars,
        name="ee_pose_err_fn",
        autograd_mode="dense",
    )
    objective = th.Objective()
    objective.add(cost_function)
    objective.to(device)

    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=10,
        step_size=5e-2,
    )
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(device)

    # Optimize
    theseus_inputs = {
        "ee_poses": ee_poses,
        "ee_poses_target": ee_poses_target,
    }

    # TODO: plot the ground truth pose along with the optimization
    # TODO: plot the URDF instead of the lineset
    # TODO: Speed and efficiency optimizations
    with torch.set_grad_enabled(True):
        for i in range(20):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            torch.cuda.reset_peak_memory_stats()
            _, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": True,
                    "track_error_history": True,
                    "damping": 0.1,
                },
            )
            end_event.record()

            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)
            log.info(f"Forward pass {i} took {forward_time} ms.")
            log.info(
                "---------------------------------------------------------------"
                "---------------------------"
            )
            theseus_inputs["ee_poses"] = (
                info.best_solution["ee_poses"].squeeze().to(device)
            )

            # Visualize robot before optimization
            if visualize:
                est_poses[tip_ids] = theseus_inputs["ee_poses"]
                viz_robot(est_poses, gt_poses, all_links, tip_ids, i)
    if visualize:
        images_to_video("output/", "fk_opt", length=10)


if __name__ == "__main__":
    visualize = True
    main(visualize)
