# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import theseus as th
from theseus.embodied.robot.forward_kinematics import Robot
import hydra
import numpy as np

import pybullet as p
import pybulletX as px

from helpers import viz_robot, images_to_video, load_robot, get_inverse_kinematics

device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
# Load the robot model and forward kinematics
URDF_PATH = "/home/suddhu/Projects/tac_neural/tac_neural/config/allegro_hand_description/allegro_hand_description_left_digit.urdf"
robot, fkin, all_links, tip_ids = load_robot(URDF_PATH)

# Given N noisy ee_poses and less-noisy encoder values, optimize for the fk_cost
# Define cost (distance between desired and current ee pose)
def ee_pose_err_fn(optim_vars, aux_vars):
    ee_poses = optim_vars[0]
    (theta_target,) = aux_vars

    poses = fkin(theta_target.tensor)
    ee_poses_target = th.SE3(tensor=torch.vstack([poses[i] for i in tip_ids]))
    pose_err = ee_poses_target.local(ee_poses)
    return pose_err


@hydra.main(config_path=".", config_name="allegro_hand")
def main(cfg):
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
    # allegro = px.Body(**cfg.allegro)

    allegro_id = p.loadURDF(cfg.allegro.urdf_path)

    # j = allegro.get_joint_states(np.arange(16))

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    n_links = len(all_links)
    NUM_DOFS = 16

    # Intialize ground truth link poses
    encoder_vals = th.Vector(tensor=torch.zeros(1, NUM_DOFS), name="encoder_vals")
    gt_poses = fkin(encoder_vals.tensor)
    gt_poses = torch.vstack(gt_poses)
    gt_poses = th.SE3(tensor=gt_poses, name="poses", dtype=torch.float32)

    # Generate noisy observed poses
    translation_noise, rotation_noise = 1e-10, 1e-10
    noise = th.SE3.exp_map(
        torch.cat(
            [
                translation_noise * (2.0 * torch.rand(n_links, 3) - 1),
                rotation_noise * (2 * torch.rand(n_links, 3) - 1),
            ],
            dim=1,
        )
    )
    noisy_poses = th.compose(gt_poses, noise)

    # Set up optimization
    ee_poses = th.SE3(tensor=noisy_poses[tip_ids], name="ee_poses")
    optim_vars = (ee_poses,)
    aux_vars = (encoder_vals,)

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
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=10,
        step_size=1e-1,
    )
    theseus_optim = th.TheseusLayer(optimizer)

    # Optimize
    theseus_inputs = {
        "ee_poses": ee_poses,
        "theta": encoder_vals,
    }

    # TODO: plot the ground truth pose along with the optimization
    # TODO: plot the URDF instead of the lineset
    # TODO: Speed and efficiency optimizations
    with torch.set_grad_enabled(True):
        for i in range(10):
            print(i)
            _, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": True,
                    "track_error_history": True,
                    "damping": 0.1,
                },
            )
            theseus_inputs["ee_poses"] = info.best_solution["ee_poses"].squeeze()

            gt_poses[tip_ids] = info.best_solution["ee_poses"].squeeze()
            # Visualize robot before optimization
            # viz_robot(gt_poses, all_links, tip_ids, i)

            get_inverse_kinematics(fkin, gt_poses)
            p.setJointMotorControlArray(
                allegro_id,
                range(16),
                p.POSITION_CONTROL,
                targetPositions=jointPoses,
            )

    images_to_video("video/", "fk_opt", length=5)


if __name__ == "__main__":
    main()
