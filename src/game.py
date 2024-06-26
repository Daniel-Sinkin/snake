import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias

import glm
import moderngl as mgl
import numpy as np
from glm import mat4, vec3
from moderngl import Buffer, Context, Texture, VertexArray
from PIL import Image

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame as pg  # noqa: E402

GRID_POSITION: TypeAlias = tuple[int, int]
SCREEN_POSITION: TypeAlias = tuple[int, int]
GAME_POSITION: TypeAlias = tuple[int, int]


@dataclass
class Settings:
    WINDOW_SIZE: tuple[int, int] = (600, 600)
    GAME_SIZE: tuple[int, int] = (600, 600)
    GRID_SIZE: int = 32
    GAMETICK_INTERVAL_BASE: float = 0.10  # Seconds between each game tick
    GAMETICK_INTERVAL_MIN: float = 0.04  # Minimum seconds between each game tick
    # Number of bodies before interval min is reached
    GAMETICK_N_BODIES_TO_INTERVAL_MIN: int = 100


class MoveDirection(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


class GraphicsEngine:
    def __init__(self):
        self.window_size: tuple[int, int] = Settings.WINDOW_SIZE
        self.aspect_ratio: float = self.window_size[0] / self.window_size[1]
        self.game_size: tuple[int, int] = Settings.GAME_SIZE

        # TODO: Find a better name for those
        self.adjust_x: float = self.game_size[0] / self.window_size[0]
        self.adjust_y: float = self.game_size[1] / self.window_size[1]
        self.adjust_vec = glm.vec2(self.adjust_x, self.adjust_y)

        pg.init()
        pg.mixer.init()

        self.pickup_sound = pg.mixer.Sound("data/pickup.mp3")

        self.grid_size: int = Settings.GRID_SIZE

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)

        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE
        )

        self.pg_window: pg.Surface = pg.display.set_mode(
            self.window_size, flags=pg.OPENGL | pg.DOUBLEBUF
        )

        self.ctx: Context = mgl.create_context()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)

        self.clock = pg.time.Clock()
        self.time = 0.0
        self.delta_time = 0

        self.is_running = True
        self.frame_counter = 0

        # INIT VBOQuad
        # fmt: off
        quad_vertices = np.array(
            [
                -1.0, -1.0,
                1.00, -1.0,
                -1.0, 1.00,
                1.00, 1.00,
            ],
            dtype=np.float32,
        )
        # fmt: on
        self.quad_vbo: Buffer = self.ctx.buffer(quad_vertices)

        board_vertex_shader = """
        # version 330

        layout(location = 0) in vec2 in_position;

        uniform mat4 m_model;

        out vec2 v_position;

        void main() {
            gl_Position = m_model * vec4(in_position, 0.0, 1.0);
            v_position = in_position;
        }
        """

        board_fragment_shader = """
        #version 330

        const float PI = 3.1415926;

        in vec2 v_position;
        out vec4 out_color;

        uniform int i_gridsize;
        // uniform float f_time;

        void main() {
            vec2 st = v_position / 2.0;

            vec2 floored = vec2(floor(st.x * i_gridsize), floor(st.y * i_gridsize));
            float f_sum = floored.x + floored.y;

            vec4 grid_color;
            if (mod(f_sum, 2.0) >= 1.0) {
                grid_color = vec4(0.2, 0.2, 0.6, 1.0);
            } else {
                grid_color = vec4(0.3, 0.3, 0.4, 1.0);
            }
            out_color = grid_color;
        }
        """
        self.board_shader_program = self.ctx.program(
            vertex_shader=board_vertex_shader, fragment_shader=board_fragment_shader
        )

        board_m_model: mat4 = mat4()
        board_m_model: mat4 = glm.scale(
            board_m_model, vec3(1 / self.aspect_ratio, 1.0, 1.0)
        )

        self.board_shader_program["m_model"].write(board_m_model)
        self.board_shader_program["i_gridsize"] = self.grid_size

        self.board_vao: VertexArray = self.ctx.vertex_array(
            self.board_shader_program, [(self.quad_vbo, "2f", "in_position")]
        )

        self.quad_vertex_shader = """
        # version 330

        layout(location = 0) in vec2 in_position;

        uniform mat4 m_model_scale;
        uniform mat4 m_model_translate;

        out vec2 v_position;

        void main() {
            gl_Position = m_model_translate * m_model_scale * vec4(in_position, 0.0, 1.0);
            v_position = in_position;
        }
        """

        self.player_fragment_shader = """
        #version 330

        const float PI = 3.1415926;

        in vec2 v_position;
        out vec4 out_color;

        uniform float f_time;
        uniform float f_time_of_last_pickup;

        void main() {
            if(f_time - f_time_of_last_pickup >= 0.5) {
                out_color = vec4(0.5 - 0.3 * abs(sin(f_time * PI * 3)), 0.7, 0.3, 1.0);
            } else {
                out_color = vec4(0.8, 0.1, 0.8, 1.0);
            
            }
        }
        """
        self.player_head_shader_program: mgl.Program = self.ctx.program(
            vertex_shader=self.quad_vertex_shader,
            fragment_shader=self.player_fragment_shader,
        )

        # self.player_base_position = vec3(-9.675, 7.0, 0.0)

        self.grid_00_position = vec3(
            -1.0 + self.adjust_x / self.grid_size,
            1.0 - self.adjust_y / self.grid_size,
            0.0,
        )
        self.player_m_model_translate: mat4 = glm.translate(self.grid_00_position)
        self.player_m_model_scale: mat4 = glm.scale(
            mat4(), vec3(1.0 / self.aspect_ratio, 1.0, 1.0)
        )
        self.player_m_model_scale = glm.scale(
            self.player_m_model_scale,
            vec3(1 / self.grid_size, 1.0 / self.grid_size, 1.0),
        )
        self.player_head_shader_program["m_model_translate"].write(
            self.player_m_model_translate
        )
        self.player_head_shader_program["m_model_scale"].write(
            self.player_m_model_scale
        )
        self.player_head_shader_program["f_time"] = 0.0
        self.player_head_shader_program["f_time_of_last_pickup"] = -1000.0

        self.player_head_vao: VertexArray = self.ctx.vertex_array(
            self.player_head_shader_program, [(self.quad_vbo, "2f", "in_position")]
        )

        self.quad_fragment_shader = """
        #version 330

        const float PI = 3.1415926;

        in vec2 v_position;
        out vec4 out_color;

        uniform float f_time;
        uniform int id;

        void main() {
            out_color = vec4(0.3 + 0.1 * sin(f_time * PI + id * PI / 6), 0.4, 0.2, 1.0);
        }
        """

        self.time_of_last_move: float = time.perf_counter()

        self.game_tick_counter = 0

        self.pickup_vertex_shader = """
        # version 330

        layout(location = 0) in vec2 in_position;

        uniform mat4 m_model_scale;
        uniform mat4 m_model_translate;
        uniform mat4 m_model_rotate;

        out vec2 v_position;

        void main() {
            gl_Position =  m_model_translate * m_model_scale * m_model_rotate * vec4(in_position, 0.0, 1.0);
            v_position = in_position;
        }
        """

        self.pickup_fragment_shader = """
        #version 330

        const float PI = 3.1415926;

        in vec2 v_position;
        out vec4 out_color;

        uniform float f_time;

        void main() {
            out_color = vec4(1.0, 0.2, 0.7 + 0.1 * sin(f_time * PI), 1.0);
        }
        """

        # fmt: off
        triangle_vertices = np.array(
            [
                -1.0, 0.5,
                0.80, -1.0,
                0.00, 1.00,
            ],
            dtype=np.float32,
        )
        self.triangle_vbo: Buffer = self.ctx.buffer(triangle_vertices)
        # fmt: on
        self.pickup_grid_position: GRID_POSITION = (5, 5)

        self.pickup_shader_program: mgl.Program = self.ctx.program(
            vertex_shader=self.pickup_vertex_shader,
            fragment_shader=self.pickup_fragment_shader,
        )

        self.pickup_shader_program["f_time"] = 0.0
        self.pickup_shader_program["m_model_scale"].write(
            glm.scale(
                mat4(),
                0.4
                * vec3(
                    1 / self.grid_size * 1 / self.aspect_ratio, 1 / self.grid_size, 1.0
                ),
            )
        )
        self.pickup_shader_program["m_model_translate"].write(
            glm.translate(vec3(self.adjust_vec * self.pickup_grid_position, 0.0))
        )

        self.pickup_vao: VertexArray = self.ctx.vertex_array(
            self.pickup_shader_program, [(self.triangle_vbo, "2f", "in_position")]
        )

        self.gametick_interval: float = Settings.GAMETICK_INTERVAL_BASE

        self.menu = Menu(self)
        self.menu.is_active = True

        self.game_running = False

    def get_shader_programs(self) -> list[mgl.Program]:
        return [
            self.board_shader_program,
            self.player_head_shader_program,
            self.pickup_shader_program,
        ]

    def _check_event_keydown_wasd(self, event: pg.event.Event) -> None:
        new_move_direction = MoveDirection.NONE
        match event.key:
            case pg.K_w:
                if self.previous_move_direction != MoveDirection.DOWN:
                    new_move_direction = MoveDirection.UP
            case pg.K_s:
                if self.previous_move_direction != MoveDirection.UP:
                    new_move_direction = MoveDirection.DOWN
            case pg.K_a:
                if self.previous_move_direction != MoveDirection.RIGHT:
                    new_move_direction = MoveDirection.LEFT
            case pg.K_d:
                if self.previous_move_direction != MoveDirection.LEFT:
                    new_move_direction = MoveDirection.RIGHT
            case _:
                raise RuntimeError("Invalid key pressed")

        if (
            new_move_direction != MoveDirection.NONE
            and new_move_direction != self.player_move_direction
        ):
            self.player_move_direction = new_move_direction

    def _check_event_keydown(self, event: pg.event.Event) -> None:
        if event.key in [pg.K_w, pg.K_s, pg.K_a, pg.K_d]:
            self._check_event_keydown_wasd(event)
        elif event.key == pg.K_SPACE:
            if self.menu.is_active:
                self.start_game()
        elif event.key == pg.K_ESCAPE:
            if self.game_running:
                self.open_main_menu()

    def check_event(self) -> None:
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    self.is_running = False
                case pg.KEYDOWN:
                    self._check_event_keydown(event)

    def _update_gamestate_moving(self) -> None:
        self.previous_move_direction = self.player_move_direction
        self.player_previous_grid_positions.append(self.player_grid_position)
        match self.player_move_direction:
            case MoveDirection.UP:
                self.player_grid_position = (
                    self.player_grid_position[0],
                    self.player_grid_position[1] - 1,
                )
            case MoveDirection.DOWN:
                self.player_grid_position = (
                    self.player_grid_position[0],
                    self.player_grid_position[1] + 1,
                )
            case MoveDirection.LEFT:
                self.player_grid_position = (
                    self.player_grid_position[0] - 1,
                    self.player_grid_position[1],
                )
            case MoveDirection.RIGHT:
                self.player_grid_position = (
                    self.player_grid_position[0] + 1,
                    self.player_grid_position[1],
                )
            case MoveDirection.NONE:
                pass

    def _update_gamestate_death_checks(self) -> None:
        if (self.player_grid_position[0] < 0) or (
            self.player_grid_position[0] >= self.grid_size
        ):
            self.on_death("Horizontal out of bounds!")
            return
        if (self.player_grid_position[1] < 0) or (
            self.player_grid_position[1] >= self.grid_size
        ):
            self.on_death("Vertical out of bounds!")
            return

        if self.player_grid_position in self.tail_positions:
            self.on_death("Collided with own tail!")
            return

    def _update_gamestate_pickup(self) -> None:
        self.player_body_parts.append(PlayerBodyPart(self))
        blocked_grids: list[GRID_POSITION] = [
            self.player_grid_position
        ] + self.tail_positions

        if len(self.player_body_parts) == self.grid_size**2:
            self.on_death("You win!")
            return

        generated_square: GRID_POSITION = self.player_grid_position
        assert (
            generated_square in blocked_grids
        ), "generated_squares should be initialized to a blocked grid."
        while generated_square in blocked_grids:
            generated_square: GRID_POSITION = (
                int(np.random.randint(self.grid_size)),
                int(np.random.randint(self.grid_size)),
            )
            print(f"{generated_square=}")
        self.pickup_grid_position = generated_square

        N = Settings.GAMETICK_N_BODIES_TO_INTERVAL_MIN
        kp = int(min(len(self.player_body_parts), N))

        gt_min = float(Settings.GAMETICK_INTERVAL_MIN)
        gt_base = float(Settings.GAMETICK_INTERVAL_BASE)
        self.gametick_interval = kp / N * gt_min + (N - kp) / N * gt_base

        self.player_head_shader_program["f_time_of_last_pickup"] = self.time

        self.pickup_sound.play()

    def update_gamestate(self) -> None:
        """
        Ticks for the actual game logic, invoked by the game loop.
        """
        if len(self.player_body_parts) == 0:
            self.tail_positions: list[GRID_POSITION] = []
        else:
            self.tail_positions: list[GRID_POSITION] = (
                self.player_previous_grid_positions[-len(self.player_body_parts) :]
            )

        self._update_gamestate_moving()

        self._update_gamestate_death_checks()

        if self.player_grid_position == self.pickup_grid_position:
            self._update_gamestate_pickup()

        self.game_tick_counter += 1

    def _update_player(self) -> None:
        for body_parts in self.player_body_parts:
            body_parts.update()

        self.player_body_m_model_translate = glm.translate(
            self.grid_00_position
            + (2 / self.grid_size)
            * vec3(
                self.adjust_vec * glm.vec2(self.player_grid_position),
                0.0,
            )
            * vec3(1.0, -1.0, 0.0)
        )
        self.player_head_shader_program["m_model_translate"].write(
            self.player_body_m_model_translate
        )

    def _update_pickup(self):
        self.pickup_body_m_model_translate = glm.translate(
            self.grid_00_position
            + (2 / self.grid_size)
            * vec3(
                self.adjust_vec * glm.vec2(self.pickup_grid_position),
                0.0,
            )
            * vec3(1.0, -1.0, 0.0)
        )
        self.pickup_shader_program["m_model_translate"].write(
            self.pickup_body_m_model_translate
        )
        self.pickup_shader_program["m_model_rotate"].write(
            glm.rotate(
                self.time * 5.0,
                vec3(0.0, 0.0, 1.0),
            ),
        )

    def update(self) -> None:
        """
        This handles all the rendering logic, the actual gameticks are handled in update_gamestate.
        """

        if self.game_running:
            assert not self.menu.is_active
            t_curr = time.perf_counter()
            if self.game_running and (
                t_curr > self.time_of_last_move + self.gametick_interval
            ):
                print(f"{self.gametick_interval=}")
                self.update_gamestate()

                self.time_of_last_move = time.perf_counter()
                print("Next move!")

            self._update_player()
            self._update_pickup()

            for program in self.get_shader_programs():
                self.player_head_shader_program["f_time"] = self.time
                self.pickup_shader_program["f_time"] = self.time
        if self.menu.is_active:
            assert not self.game_running
            self.menu.update()

    def render(self) -> None:
        self.ctx.clear(color=(1.0, 0.0, 1.0))

        if self.menu.is_active:
            assert not self.game_running

            self.menu.render()
        elif self.game_running:
            self.pickup_vao.render(mgl.LINE_LOOP)
            self.player_head_vao.render(mgl.TRIANGLE_STRIP)
            for body_part in self.player_body_parts:
                body_part.render()

            self.board_vao.render(mgl.TRIANGLE_STRIP)
        else:
            raise RuntimeError("Neither menu nor game is active!")

        pg.display.flip()

    def iteration(self) -> None:
        self.time: float = pg.time.get_ticks() / 1000.0

        self.update()
        self.render()

    def run(self) -> None:
        while self.is_running:
            self.check_event()
            self.iteration()

            self.clock.tick(60.0)

    def on_death(self, info=None) -> None:
        self.open_main_menu()

    def open_main_menu(self) -> None:
        self.menu.is_active = True
        self.game_running = False

    def setup_gamestart(self) -> None:
        self.player_grid_position: GRID_POSITION = (
            self.grid_size // 2,
            self.grid_size // 2,
        )
        self.player_previous_grid_positions: list[GRID_POSITION] = [
            (self.player_grid_position[0], self.player_grid_position[1] - i)
            for i in range(1, 6)
        ][::-1]

        self.blocked_grids: list[GRID_POSITION] = (
            self.player_previous_grid_positions
            + [
                self.player_grid_position,
            ]
        )
        for p1, p2 in zip(self.blocked_grids[:-1], self.blocked_grids[1:]):
            # Not equal and exactly one vertical or horizontal step
            assert abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

        # No duplicates
        assert len(self.player_previous_grid_positions) == len(
            set(self.player_previous_grid_positions)
        )

        self.player_move_direction = MoveDirection.DOWN
        self.previous_move_direction = MoveDirection.DOWN

        PlayerBodyPart._id = 0

        self.player_body_parts: list[PlayerBodyPart] = [
            PlayerBodyPart(self)
            for _ in range(len(self.player_previous_grid_positions))
        ]

    def start_game(self) -> None:
        assert self.menu.is_active
        self.menu.is_active = False

        self.setup_gamestart()

        self.game_running = True

    def get_texture(self) -> Texture:
        image: Image.Image = Image.open(
            "/Users/danielsinkin/GitHub_private/snake/data/texture.jpg"
        )
        image = image.convert("RGB")

        data: bytes = np.array(image).tobytes()

        texture: Texture = self.ctx.texture(size=image.size, components=3, data=data)

        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()

        texture.anisotropy = 16.0

        return texture


class Pickup:
    def __init__(self, app: GraphicsEngine) -> None:
        self.app: GraphicsEngine = app
        self.ctx: Context = self.app.ctx
        self.position: GAME_POSITION = (0, 0)


class PlayerBodyPart:
    _id = 0

    def __init__(self, app: GraphicsEngine):
        self.id = PlayerBodyPart._id
        PlayerBodyPart._id += 1

        self.app: GraphicsEngine = app
        self.ctx: Context = self.app.ctx
        self.program: mgl.Program = self.ctx.program(
            vertex_shader=self.app.quad_vertex_shader,
            fragment_shader=self.app.quad_fragment_shader,
        )
        self.start_position = vec3(0.0, 0.0, 0.0)
        self.m_model_translate: mat4 = glm.translate(self.start_position)
        self.m_model_scale: mat4 = glm.scale(
            mat4(), vec3(1.0 / self.app.aspect_ratio, 1.0, 1.0)
        )
        self.m_model_scale: mat4 = glm.scale(
            self.m_model_scale,
            vec3(1 / self.app.grid_size, 1.0 / self.app.grid_size, 1.0),
        )
        self.m_model_scale: mat4 = glm.scale(
            self.m_model_scale, vec3(1.0 - 0.2 * min(self.id, 3))
        )
        self.program["m_model_translate"].write(self.m_model_translate)
        self.program["m_model_scale"].write(self.m_model_scale)
        self.program["f_time"] = 0.0
        self.program["id"] = self.id

        self.vao: VertexArray = self.ctx.vertex_array(
            self.program, [(self.app.quad_vbo, "2f", "in_position")]
        )

    def update(self) -> None:
        m_model_translate: mat4 = glm.translate(
            self.app.grid_00_position
            + (2 / self.app.grid_size)
            * vec3(
                self.app.adjust_vec
                * glm.vec2(self.app.player_previous_grid_positions[-(self.id + 1)]),
                0.0,
            )
            * vec3(1.0, -1.0, 0.0)
        )
        self.program["m_model_translate"].write(m_model_translate)
        self.program["f_time"] = self.app.time

    def render(self) -> None:
        self.vao.render(mgl.TRIANGLE_STRIP)

    def __del__(self) -> None:
        try:
            if self.program is not None:
                self.program.release()
        except AttributeError:
            pass
        try:
            if self.vao is not None:
                self.vao.release()
        except AttributeError:
            pass


class Menu:
    def __init__(self, app: GraphicsEngine):
        self.app: GraphicsEngine = app
        self.ctx: Context = self.app.ctx

        vertex_shader = """
        # version 330

        layout(location = 0) in vec2 in_position;

        out vec2 v_position;

        void main() {
            v_position = in_position;
            gl_Position = vec4(v_position, 0.0, 1.0);
        }
        """

        fragment_shader = """
        # version 330

        const float PI = 3.1415926;

        out vec4 out_color;
        
        uniform vec2 u_resolution;
        uniform float u_time;

        uniform vec2 u_mouse;

        void main() {
           float aspect = u_resolution.x / u_resolution.y;    
    
            vec2 st = gl_FragCoord.xy / u_resolution.xy;

            float t = u_time;

            vec2 smouse = u_mouse.xy / u_resolution.xy;
            smouse.y = 1.0 - smouse.y;

            float r1 = 0.1 + 0.05 * sin(PI * u_time); // radius for the first ring
            float r2 = 0.3 + 0.05 * sin(PI * u_time); // radius for the second ring
            vec2 points1[6];
            vec2 points2[6];

            for (int i = 0; i < 6; i++) {
                float angle1 = 2.0 * PI * float(i) / 6.0 + PI * u_time;
                float angle2 = 2.0 * PI * float(i) / 6.0 - PI * u_time;
                points1[i] = smouse + r1 * vec2(cos(angle1) / aspect, sin(angle1));
                points2[i] = smouse + r2 * vec2(cos(angle2) / aspect, sin(angle2));
            }

            float product = distance(smouse, st);
            for (int i = 0; i < 6; i++) {
                product *= distance(points1[i], st);
                product *= distance(points2[i], st);
            }

            vec3 color = vec3(0.0);
            float val = max(1 - pow(product, 0.03 + 0.02 * sin(u_time * PI)), 0.0);
            color = 0.8 * vec3(val, 0.0, val);

            out_color = vec4(color, 1.0);
        }
        """
        self.program: mgl.Program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        self.program["u_resolution"] = self.app.window_size
        self.program["u_time"] = self.app.time
        self.program["u_mouse"] = pg.mouse.get_pos()

        self.vao: VertexArray = self.ctx.vertex_array(
            self.program, [(self.app.quad_vbo, "2f", "in_position")]
        )

        self.is_active = False

    def update(self) -> None:
        if self.is_active:
            self.program["u_time"] = self.app.time
            self.program["u_mouse"] = pg.mouse.get_pos()

    def render(self):
        self.vao.render(mgl.TRIANGLE_STRIP)
