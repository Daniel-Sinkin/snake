import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
from abc import ABC, abstractmethod
from enum import Enum, auto

import glm
import moderngl as mgl
import numpy as np
import pygame as pg
from moderngl import Context

from .math import ndc_to_screenspace, screenspace_to_ndc


class MoveDirection(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


class GraphicsEngine:
    def __init__(self):
        self.window_size: tuple[int, int] = (800, 600)
        self.aspect_ratio: float = self.window_size[0] / self.window_size[1]
        pg.init()

        self.grid_size = 8

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)

        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE
        )

        self.pg_window = pg.display.set_mode(
            self.window_size, flags=pg.OPENGL | pg.DOUBLEBUF
        )

        self.ctx = mgl.create_context()
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
        quad_vbo = self.ctx.buffer(quad_vertices)
        buffer_format = "2f"
        attributes = ["in_position"]

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
        uniform float f_time;

        void main() {
            vec2 st = v_position / 2.0;

            vec2 floored = vec2(floor(st.x * i_gridsize), floor(st.y * i_gridsize));
            float f_sum = floored.x + floored.y;

            if (mod(f_sum, 2.0) >= 1.0) {
                out_color = vec4(0.2 * (2 + sin(PI * f_time * 0.5)), vec2(0.0), 1.0);
            } else {
                out_color = vec4(0.2 * (2 + sin(PI * f_time * 0.5 + PI)), vec2(0.0), 1.0);
            }
        }
        """
        self.board_shader_program = self.ctx.program(
            vertex_shader=board_vertex_shader, fragment_shader=board_fragment_shader
        )

        board_m_model = glm.translate(glm.vec3(-0.25, 0.0, 0.0))
        board_m_model = glm.scale(
            board_m_model, glm.vec3(1 / self.aspect_ratio, 1.0, 1.0)
        )

        self.board_shader_program["m_model"].write(board_m_model)
        self.board_shader_program["i_gridsize"] = self.grid_size

        self.board_vao = self.ctx.vertex_array(
            self.board_shader_program, [(quad_vbo, "2f", "in_position")]
        )

        player_vertex_shader = """
        # version 330

        layout(location = 0) in vec2 in_position;

        uniform mat4 m_model_scale;
        uniform mat4 m_model_translate;

        out vec2 v_position;

        void main() {
            gl_Position = m_model_scale * m_model_translate * vec4(in_position, 0.0, 1.0);
            v_position = in_position;
        }
        """

        player_fragment_shader = """
        #version 330

        const float PI = 3.1415926;

        in vec2 v_position;
        out vec4 out_color;

        uniform float f_time;

        void main() {
            out_color = vec4(0.5 - 0.3 * abs(sin(f_time * PI * 3)), 0.7, 0.3, 1.0);
        }
        """
        self.player_shader_program = self.ctx.program(
            vertex_shader=player_vertex_shader, fragment_shader=player_fragment_shader
        )

        self.player_m_model_translate = glm.translate(glm.vec3(-9.675, 7.0, 0.0))
        self.player_m_model_scale = glm.scale(
            glm.mat4(), glm.vec3(1 / self.aspect_ratio, 1.0, 1.0)
        )
        self.player_m_model_scale = glm.scale(
            self.player_m_model_scale,
            glm.vec3(1 / self.grid_size, 1 / self.grid_size, 1.0),
        )
        self.player_shader_program["m_model_translate"].write(
            self.player_m_model_translate
        )
        self.player_shader_program["m_model_scale"].write(self.player_m_model_scale)
        self.player_shader_program["f_time"] = 0.0

        self.player_vao = self.ctx.vertex_array(
            self.player_shader_program, [(quad_vbo, "2f", "in_position")]
        )

        self.player_grid_position = (0, 0)
        self.move_direction = MoveDirection.NONE
        self.previous_move = MoveDirection.NONE

    def check_event(self) -> None:
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    self.is_running = False
                case pg.KEYDOWN:
                    if event.key in [pg.K_w, pg.K_s, pg.K_a, pg.K_d]:
                        match event.key:
                            case pg.K_w:
                                if self.previous_move != MoveDirection.DOWN:
                                    self.move_direction = MoveDirection.UP
                            case pg.K_s:
                                if self.previous_move != MoveDirection.UP:
                                    self.move_direction = MoveDirection.DOWN
                            case pg.K_a:
                                if self.previous_move != MoveDirection.RIGHT:
                                    self.move_direction = MoveDirection.LEFT
                            case pg.K_d:
                                if self.previous_move != MoveDirection.LEFT:
                                    self.move_direction = MoveDirection.RIGHT

    def update_gamestate(self):
        if self.move_direction == MoveDirection.UP:
            self.player_grid_position = (
                self.player_grid_position[0],
                self.player_grid_position[1] + 1,
            )
        elif self.move_direction == MoveDirection.DOWN:
            self.player_grid_position = (
                self.player_grid_position[0],
                self.player_grid_position[1] - 1,
            )
        elif self.move_direction == MoveDirection.LEFT:
            self.player_grid_position = (
                self.player_grid_position[0] - 1,
                self.player_grid_position[1],
            )
        elif self.move_direction == MoveDirection.RIGHT:
            self.player_grid_position = (
                self.player_grid_position[0] + 1,
                self.player_grid_position[1],
            )

        self.previous_move: MoveDirection = self.move_direction

    def update(self) -> None:
        print(self.move_direction)
        self.player_shader_program["m_model_translate"].write(
            self.player_m_model_translate
        )

        self.board_shader_program["f_time"] = self.time
        self.player_shader_program["f_time"] = self.time

    def render(self) -> None:
        self.ctx.clear(color=(0.08, 0.16, 0.18))

        self.player_vao.render(mgl.TRIANGLE_STRIP)
        self.board_vao.render(mgl.TRIANGLE_STRIP)

        pg.display.flip()

    def iteration(self) -> None:
        self.time = pg.time.get_ticks() / 1000.0

        self.update()
        self.render()

        self.delta_time = self.clock.tick(60.0) / 1000.0

    def run(self) -> None:
        while self.is_running:
            self.check_event()
            self.iteration()

            self.frame_counter += 1
