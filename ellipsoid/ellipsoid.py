import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def ellipsoid(rx, ry, rz, s1, s2):
    vertices, indices, color = [], [], []
    for i in range(s1 + 1):
        phi = np.pi / 2 - np.pi * i / s1
        for j in range(s2 + 1):
            theta = 2 * np.pi * j / s2
            x = rx * np.cos(phi) * np.cos(theta)
            y = ry * np.cos(phi) * np.sin(theta)
            z = rz * np.sin(phi)
            vertices += [[x, y, z]]
            if i % 2 == 0 and j % 2 == 0:
                color += [1, 0, 0]
            elif i % 2 == 0 and j % 2 != 0:
                color += [0, 0, 1]
            elif i % 2 != 0 and j % 2 == 0:
                color += [0, 0, 1]
            elif i % 2 != 0 and j % 2 != 0:
                color += [1, 0, 0]

    for i in range(s1):
        k1 = i * (s2 + 1)
        k2 = k1 + s2 + 1
        for j in range(s2):
            if i != 0:
                indices += [k1, k2, k1 + 1]
            if i != (s1 - 1):
                indices += [k1 + 1, k2, k2 + 1]
            k1 += 1
            k2 += 1

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    color = np.array(color, dtype=np.float32)
    
    return vertices, indices, color

class Ellipsoid(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors = ellipsoid(1, 1.5, 2, 100, 100) # x_radius, y_radius, z_radius, stack, sector
        
        # self.normals = [] # YOUR CODE HERE to compute vertex's normal using the coordinates
        
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
