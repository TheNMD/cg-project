import ctypes
import glfw
import os, sys
import numpy as np
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from libs.shader import *
from libs import transform as T
from libs.buffer import *

class Cube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array(
            # YOUR CODE HERE to specify vertex's coordinates
            [
                [-1.0, -1.0,  1.0], # 0
                [ 1.0, -1.0,  1.0], # 1
                [ 1.0,  1.0,  1.0], # 2
                [-1.0,  1.0,  1.0], # 3
                [-1.0, -1.0, -1.0], # 4
                [ 1.0, -1.0, -1.0], # 5
                [ 1.0,  1.0, -1.0], # 6
                [-1.0,  1.0, -1.0]  # 7
            ], dtype = np.float32
        )

        self.indices = np.array(
            # YOUR CODE HERE to specify index data
            [
                7, 6, 4, 5, 1, 6, 2, 7, 3, 4, 0, 1, 3, 2
            ], dtype = np.uint32
        )

        self.normals = [] # YOUR CODE HERE to compute vertex's normal using the coordinates

        # colors: RGB format
        self.colors = np.array(
            # YOUR CODE HERE to specify vertex's color
            [
                [1, 0, 0], # 0
                [1, 0, 1], # 1
                [1, 1, 1], # 2
                [1, 1, 0], # 3
                [0, 0, 0], # 4
                [0, 0, 1], # 5
                [0, 1, 1], # 6
                [0, 1, 0]  # 7
            ],dtype= np.float32
        )

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #
     

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

