import abc
import os
import math

import numpy as np
import genesis as gs
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))

stl_dir = os.path.join(script_dir, 'stl')

goal_path = os.path.join(stl_dir, 'goal.stl')
wall_path = os.path.join(stl_dir, 'wall.stl')
line_path = os.path.join(stl_dir, 'line.stl')

gs.init(
    seed=None,
    precision='64',
    debug=False,
    eps=1e-12,
    logging_level=None,
    backend=gs.cpu,
    theme='dark',
    logger_verbose_time=False
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
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=True,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)
#
plane = scene.add_entity(gs.morphs.Plane())
# wall = scene.add_entity(
#     gs.morphs.Mesh(
#         file=wall_path,
#         scale=(0.001, 0.001, 0.001),
#         pos=(0.0, 1.0, 0.0),
#         euler=(90.0, 0.0, 270.0),
#         convexify=True,
#         decimate=False,
#         visualization=True,
#         collision=True,
#     ),
# )
goal = scene.add_entity(
    gs.morphs.Mesh(
        file=goal_path,
        scale=(0.001, 0.001, 0.001),
        pos=(0.0, 1.0, 0.0),
        euler=(90.0, 0.0, 270.0),
        convexify=True,
        decimate=False,
        visualization=True,
        collision=True,
    ),
)

scene.build()

for i in range(100000):
    scene.step()



