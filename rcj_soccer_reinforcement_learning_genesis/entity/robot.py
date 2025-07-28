import abc
import os
import math

import numpy as np
import genesis as gs
import torch

from rcj_soccer_reinforcement_learning_genesis.entity.entity import Robot


script_dir = os.path.dirname(os.path.abspath(__file__))

class Agent(Robot):
    def __init__(self, scene: gs.Scene = None, num_envs: int = 1):
        super().__init__()
        self.agent = None
        self.default_ori = [0.0, 0.0, 0.0]
        self.position = None
        self.scene = scene
        self.robot_collision_path = os.path.join(script_dir, 'stl', 'robot_v2_collision.stl')
        self.robot_visual_path = os.path.join(script_dir, 'stl', 'robot_v2_visual.stl')
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

    def create(self):

        self.agent = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file=self.robot_visual_path,
                scale=(0.0001, 0.0001, 0.0001),
                # pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                visualization=True,
                collision=True,
                convexify=True,
                decimate=False,
                fixed=False,
            ),
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="visual"
        )

        return self.agent

    def action(self, velocity_x, velocity_y, velocity_theta, envs_idx=None):
        # vel_cmd = np.stack([velocity_x, velocity_y], axis=1).astype(np.float64)
        vel_cmd = np.array([velocity_x, velocity_y, 0.0, 0.0, 0.0, velocity_theta], dtype=np.float32)

        if envs_idx is not None:
            idx = np.r_[envs_idx].tolist()
            vel_cmd = vel_cmd[idx]

        self.agent.control_dofs_velocity(vel_cmd, envs_idx)

    def reset(self, env_idx):
        env_idx = np.asarray(env_idx, dtype=np.int32)
        num_envs = len(env_idx)

        zeros = np.zeros((num_envs, 1), dtype=np.float64)
        zeros_wheel = np.zeros((num_envs, len(self.wheel_dofs)), dtype=np.float64)

        self.agent.set_dofs_position(
            zeros_wheel,
            self.wheel_dofs,
            zero_velocity=True,
            envs_idx=env_idx.tolist()
        )

        self.agent.set_dofs_velocity(
            zeros_wheel,
            self.wheel_dofs,
            envs_idx=env_idx.tolist()
        )

        self.agent.set_dofs_position(
            zeros,
            [self.pipe_dof],
            zero_velocity=True,
            envs_idx=env_idx.tolist()
        )

        self.agent.set_dofs_velocity(
            zeros,
            [self.pipe_dof],
            envs_idx=env_idx.tolist()
        )

if __name__ == "__main__":
    gs.init(
        seed = None,
        precision = '32',
        debug = False,
        eps = 1e-12,
        backend = gs.gpu,
        theme = 'dark',
        logger_verbose_time = False
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=150,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False
        ),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame = True,
            world_frame_size = 1.0,
            show_link_frame = False,
            show_cameras = False,
            plane_reflection = False,
            ambient_light = (0.1, 0.1, 0.1),
            n_rendered_envs = 100,
        ),
        renderer = gs.renderers.Rasterizer(),
    )

    cam = scene.add_camera(
        res    = (1280, 960),
        pos    = (3.5, 0.0, 2.5),
        lookat = (0, 0, 0.5),
        fov    = 30,
        GUI    = False
    )

    plane = scene.add_entity(gs.morphs.Plane())
    num = 1

    agent = Agent(scene)
    agent.create()
    scene.build(n_envs=num, env_spacing=(0.5, 0.5))
    cam.start_recording()

    for i in range(10000):
        scene.step()
        agent.action(1.0, 1.0, 1.0)