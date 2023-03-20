import ctypes
import glfw
import math
import os, sys
import numpy as np
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from libs.shader import *
from libs import transform as T
from libs.buffer import *


def create_cone(num_slices, height, radius):
    angle = (2.0 * np.pi) / num_slices
    vertices = np.zeros((num_slices + 2, 3), dtype=np.float32)

    # Create the tip of the cone
    vertices[0] = (0, height/2.0, 0)

    # Create the base of the cone
    for i in range(num_slices):
        x = radius * np.cos(i * angle)
        z = radius * np.sin(i * angle)
        vertices[i+1] = (x, -height/2.0, z)


    # Create the index buffer for the triangle strip
    indices = np.zeros(num_slices * 3 + 3, dtype=np.uint32)
    index = 0
    for i in range(num_slices):
        indices[index] = i + 1
        indices[index+1] = i + 2
        indices[index+2] = 0
        index += 3
    indices[num_slices * 3] = 1
    indices[num_slices * 3 + 1] = num_slices + 1
    indices[num_slices * 3 + 2] = 1

    return vertices, indices
class Cone(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices = create_cone(4, 2.0, 1.0) 
        self.normals = [] # YOUR CODE HERE to compute vertex's normal using the coordinates
        print(self.vertices)
        print(self.indices)
        # colors: RGB format
        self.colors = np.array(
            # YOUR CODE HERE to specify vertex's color
            [
                [0.5, 0, 0.5],
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                [1, 1, 1],
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

