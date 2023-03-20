import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

def ellipsoid(r, h, s):
    vertices, indices, color = [], [], []
    for i in range(s):
        theta = 2 * np.pi * i / s
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vertices += [[x, y, 0]]
        color += [0, 1, 0]
        
    vertices += [[0, 0, h / 2]]
    
    # Top side
    for i in range(len(vertices) - 1):
        indices += [i] + [len(vertices) - 1]
    indices += [0] + [len(vertices) - 1]
    
    # Move from top to bottom
    indices += [0] + [0]
    
    vertices += [[0, 0, -h / 2]]
    
    # Bottom side
    for i in range(len(vertices) - 2):
        indices += [i] + [len(vertices) - 1]
    indices += [0] + [len(vertices) - 1]    
    
    color += [0, 0, 1] + [1, 0, 0]
    
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    color = np.array(color, dtype=np.float32)
    
    return vertices, indices, color

class Ellipsoid(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors = ellipsoid(1, 2, 200) # radius, height, side
        
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

