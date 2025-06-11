import os

import genesis as gs

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

plane = scene.add_entity(gs.morphs.Plane())

wall = scene.add_entity(
    morph = gs.morphs.Mesh(
        file=wall_path,
        scale=(0.001, 0.001, 0.001),
        pos=(0.0, 0.0, 0.0),
        euler=(90.0, 0.0, 270.0),
        fixed=True,
        convexify=True,
        decimate=False,
        visualization=True,
        collision=True,
    ),
    surface = gs.surfaces.Default(color=(0.0, 0.0, 0.0)),
)

blue_goal = scene.add_entity(
    gs.morphs.Mesh(
        file=goal_path,
        scale=(0.001, 0.001, 0.001),
        pos=(0.62, 0.02, 0.0),
        euler=(90.0, 0.0, 270.0),
        fixed=True,
        convexify=True,
        decimate=False,
        visualization=True,
        collision=True,
        quality = True
    ),
    surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0)),
)

yellow_goal = scene.add_entity(
    gs.morphs.Mesh(
        file=goal_path,
        scale=(0.001, 0.001, 0.001),
        pos=(1.24, 2.45, 0.0),
        euler=(90.0, 0.0, 90.0),
        fixed=True,
        convexify=True,
        decimate=False,
        visualization=True,
        collision=True,
        quality = True
    ),
    surface=gs.surfaces.Default(color=(1.0, 1.0, 0.0)),
)

line = scene.add_entity(
    gs.morphs.Mesh(
        file=line_path,
        scale=(0.001, 0.001, 0.001),
        pos=(0.14, 0.14, 0.0),
        euler=(0.0, 0.0, 0.0),
        fixed=True,
        convexify=True,
        decimate=False,
        visualization=True,
        collision=True,
        quality = True
    ),
    surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
)

ball = scene.add_entity(
    gs.morphs.Sphere(
        pos=(0.0, 0.0, 0.0),
        euler = (0.0, 0.0, 0.0),
        radius=0.037,
        visualization=True,
        collision=True,
        fixed=False,
    ),
    surface=gs.surfaces.Default(color=(0.15, 0.15, 0.15)),
    entity=gs.entities.RigidEntity,
    mass=0.10
)

scene.build()

for i in range(100000):
    scene.step()

class Court(object):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.wall_position = [0.0 + self.cp[0], 0.0 + self.cp[1], 0.0 + self.cp[2]]
        self.blue_goal_position = [0.62 + self.cp[0], 0.02 + self.cp[1], 0.0]
        self.yellow_goal_position = [1.24 + self.cp[0], 2.45 + self.cp[1], 0.0]
        self.line_position = [0.14 + self.cp[0], 0.14 + self.cp[1], 0.0 + self.cp[2]]
        self.wall = None
        self.blue_goal = None
        self.yellow_goal = None
        self.line = None

    def create_court(self):
        self.wall = scene.add_entity(
            gs.morphs.Mesh(
                file=wall_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.wall_position,
                euler=(90.0, 0.0, 270.0),
                fixed=True,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
            ),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 0.0)),
        )

        self.blue_goal = scene.add_entity(
            gs.morphs.Mesh(
                file=goal_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.blue_goal_position,
                euler=(90.0, 0.0, 270.0),
                fixed=True,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
                quality=True
            ),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0)),
        )

        self.yellow_goal = scene.add_entity(
            gs.morphs.Mesh(
                file=goal_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.yellow_goal_position,
                euler=(90.0, 0.0, 90.0),
                fixed=True,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
                quality=True
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 0.0)),
        )

        self.line = scene.add_entity(
            gs.morphs.Mesh(
                file=line_path,
                scale=(0.001, 0.001, 0.001),
                pos=self.line_position,
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                convexify=True,
                decimate=False,
                visualization=True,
                collision=True,
                quality=True
            ),
        )
        return self.wall, self.blue_goal, self.yellow_goal, self.line

class Ball(object):
    def __init__(self, create_position=None):
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.ball_cp = create_position
        self.ball = None

    def create_ball(self):
        self.ball = scene.add_entity(
            gs.morphs.Sphere(
                pos=self.ball_cp,
                euler=(0.0, 0.0, 0.0),
                radius=0.037,
                visualization=True,
                collision=True,
                fixed=False,
            ),
            surface=gs.surfaces.Default(color=(0.15, 0.15, 0.15)),
            entity=gs.entities.RigidEntity,
            mass=0.10
        )
        return self.ball