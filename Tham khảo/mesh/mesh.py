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

def gen_points(smoothness):
    step =  10.0 / smoothness
    points = []
    x = 10.0
    y = 10.0
    for i in range(smoothness * 2 + 1):
        for t in range(smoothness * 2 + 1):
            points.append([x, 0, y])
            y -= step
        y = 10.0
        x -= step
    return points

def mesh(smoothness, aconst = 1.0, bconst = 1.0, const = 0.0):
    vertices, indices, color = [], [], []
    points = gen_points(smoothness)
    print(points)
    return vertices, indices, color

class Mesh(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors = mesh(5)
        self.normals = [] # YOUR CODE HERE to compute vertex's normal using the coordinates
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

