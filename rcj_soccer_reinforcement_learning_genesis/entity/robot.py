import abc
import os
import math

import numpy as np
import genesis as gs
import torch

from rcj_soccer_reinforcement_learning_genesis.entity.entity import Robot


script_dir = os.path.dirname(os.path.abspath(__file__))

stl_dir = os.path.join(script_dir, 'stl')

robot_collision_path = os.path.join(stl_dir, 'robot_v2_collision.stl')
robot_visual_path = os.path.join(stl_dir, 'robot_v2_visual.stl')

gs.init(
    seed = None,
    precision = '64',
    debug = False,
    eps = 1e-12,
    logging_level = None,
    backend = gs.cpu,
    theme = 'dark',
    logger_verbose_time = False
)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.806),
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
        show_link_frame = True,
        show_cameras = False,
        plane_reflection = True,
        ambient_light = (0.1, 0.1, 0.1),
    ),
    renderer = gs.renderers.Rasterizer(),
)
#
plane = scene.add_entity(gs.morphs.Plane())

class Agent(Robot):

    def __init__(self, create_position=None):
        super().__init__()
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.start_pos = None
        self.default_ori = [0.0, 0.0, 0.0]
        self.position = self.start_pos
        self.agent = None
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

    def create(self, position=None):

        self.position = position

        if position is None:
            self.position = self.start_pos

        self.agent = scene.add_entity(
            morph = gs.morphs.Mesh(
                file=robot_collision_path,
                scale=(0.0001, 0.0001, 0.0001),
                pos=self.cp,
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
            ),
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="visual"
        )

        return self.agent

    def action(self, robot_id, angle_deg):
        pass

agent = Agent()
agent.create()

num = 10
scene.build(n_envs=num, env_spacing=(0.5, 0.5))

for i in range(100000):
    scene.step()