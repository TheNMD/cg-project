import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

class Cube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
            [+1, +1, +1],
            [+1, +1, -1],
            [+1, -1, +1],
            [+1, -1, -1],
            [-1, +1, +1],
            [-1, +1, -1],
            [-1, -1, +1],
            [-1, -1, -1]], dtype=np.float32)

        self.indices = np.array([
            0,2,3, 
            0,1,3,  
            0,4,6, 
            0,2,6,  
            0,1,5, 
            0,4,5,
            7,5,4, 
            7,6,4,  
            7,6,2, 
            7,3,2,  
            7,5,1, 
            7,3,1],  dtype=np.uint32)

        # self.normals = # YOUR CODE HERE to compute vertex's normal using the coordinates

        # colors: RGB format
        self.colors = np.array([
            [0, 1, 1, 1], 
            [0, 0, 1, 1], 
            [0, 0, 0, 1], 
            [0, 1, 0, 1],
            [1, 1, 0, 1], 
            [1, 1, 1, 1], 
            [1, 0, 1, 1], 
            [1, 0, 0, 1]],  dtype=np.float32)

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

