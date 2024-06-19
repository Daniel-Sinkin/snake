import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
from abc import ABC, abstractmethod

import glm
import moderngl as mgl
import numpy as np
import pygame as pg
from moderngl import Context

from .math import ndc_to_screenspace, screenspace_to_ndc


class GraphicsEngine:
    def __init__(self):
        self.window_size: tuple[int, int] = (800, 600)
        self.aspect_ratio: float = self.window_size[0] / self.window_size[1]
        pg.init()

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

        vertex_shader = """
        # version 330

        layout(location = 0) in vec2 in_position;

        uniform mat4 m_model;

        out vec2 v_position;

        void main() {
            gl_Position = m_model * vec4(in_position, 0.0, 1.0);
            v_position = in_position;
        }
        """

        fragment_shader = """
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
        self.shader_program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        m_model = glm.mat4()
        m_model = glm.translate(m_model, glm.vec3(-0.25, 0.0, 0.0))
        m_model = glm.scale(m_model, glm.vec3(1 / self.aspect_ratio, 1.0, 1.0))

        self.shader_program["m_model"].write(m_model)
        self.shader_program["i_gridsize"] = 8

        self.quad_vao = self.ctx.vertex_array(
            self.shader_program, [(quad_vbo, "2f", "in_position")]
        )

    def check_event(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False

    def update(self) -> None:
        self.shader_program["f_time"] = self.time

    def render(self) -> None:
        self.ctx.clear(color=(0.08, 0.16, 0.18))

        self.quad_vao.render(mgl.TRIANGLE_STRIP)

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
