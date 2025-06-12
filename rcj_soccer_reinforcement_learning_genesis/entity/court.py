import abc
import os

import numpy as np
import genesis as gs

from rcj_soccer_reinforcement_learning_genesis.entity.entity import Entity

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
        plane_reflection=False,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
            gs.morphs.Plane(),
            material=None,
            surface=gs.surfaces.Default(
                color=(0.133, 0.545, 0.133),
                opacity=1.0,
                roughness=1.0,
                metallic=0.0,
                emissive=None
            ),
            visualize_contact=False,
            vis_mode="visual"
)

class Wall(Entity):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.wall = None
        self.surfaces = gs.surfaces.Default(
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                roughness=0.5,
                metallic=0.0,
                emissive=None)

    def create(self):
        self.wall = scene.add_entity(
            morph = gs.morphs.Mesh(
                file = wall_path,
                scale = (0.001, 0.001, 0.001),
                pos = (0.0, 0.0, 0.0),
                euler = (90.0, 0.0, 270.0),
                fixed = True,
                convexify = True,
                decimate = False,
                visualization = True,
                collision = True,
            ),
            material = None,
            surface = self.surfaces,
            visualize_contact = False,
            vis_mode = "visual"
        )
        return self.wall

class BlueGoal(Entity):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.goal = None
        self.surfaces = gs.surfaces.Default(
                                    color=(0.0, 0.0, 1.0),
                                    opacity=1.0,
                                    roughness=0.5,
                                    metallic=0.0,
                                    emissive=None)
    def create(self):
        self.goal = scene.add_entity(
            gs.morphs.Mesh(
                file=goal_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.cp,
                euler=(90.0, 0.0, 270.0),
                fixed=True,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
                quality=True
            ),
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="visual"
        )
        return self.goal

class YellowGoal(Entity):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.goal = None
        self.surfaces = gs.surfaces.Default(
                                    color=(1.0, 1.0, 0.0),
                                    opacity=1.0,
                                    roughness=0.5,
                                    metallic=0.0,
                                    emissive=None)

    def create(self):
        self.goal = scene.add_entity(
            gs.morphs.Mesh(
                file=goal_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.cp,
                euler=(90.0, 0.0, 90.0),
                fixed=True,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
                quality=True
            ),
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="visual"
        )
        return self.goal

class Line(Entity):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.line = None
        self.surfaces = gs.surfaces.Default(
                                    color=(1.0, 1.0, 1.0),
                                    opacity=1.0,
                                    roughness=0.5,
                                    metallic=0.0,
                                    emissive=None)

    def create(self):
        self.line = scene.add_entity(
            gs.morphs.Mesh(
                file=line_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.cp,
                euler=(0.0, 0.0, 0.005),
                fixed=False,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
                quality=True
            ),
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="visual"
        )
        return self.line

class Ball(Entity):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.ball_cp = create_position
        self.ball = None

    def create(self):
        self.ball = scene.add_entity(
            morph = gs.morphs.Sphere(
                pos=self.ball_cp,
                euler=(0.0, 0.0, 0.0),
                radius=0.037,
                visualization=True,
                collision=True,
                fixed=False,
            ),
            material=None,
            surface=gs.surfaces.Default(color=(0.15, 0.15, 0.15)),
            visualization_contact=False,
            vis_mode="visual"
        )
        return self.ball

class Court(object):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.wall = Wall(create_position=self.cp)
        self.blue_goal = BlueGoal(create_position=[0.62 + self.cp[0], 0.02 + self.cp[1], 0.0])
        self.yellow_goal = YellowGoal(create_position=[1.24 + self.cp[0], 2.45 + self.cp[1], 0.0])
        self.line = Line(create_position=[0.14 + self.cp[0], 0.14 + self.cp[1], 0.0])
        self.wall_entity = None
        self.blue_goal_entity = None
        self.yellow_goal_entity = None
        self.line_entity = None

    def create_court(self):
        self.wall_entity = self.wall.create()
        self.blue_goal_entity = self.blue_goal.create()
        self.yellow_goal_entity = self.yellow_goal.create()
        self.line_entity = self.line.create()
        return self.wall_entity, self.blue_goal_entity, self.yellow_goal_entity, self.line_entity


court = Court([0.0, 0.0, 0.0])
court.create_court()
scene.build()

for i in range(100000):
    scene.step()