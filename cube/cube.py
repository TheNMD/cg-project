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
                [+1, +1, +1], # A 0
                [+1, +1, -1], # B 1
                [+1, -1, +1], # C 2
                [+1, -1, -1], # D 3
                [-1, +1, +1], # E 4
                [-1, +1, -1], # F 5
                [-1, -1, +1], # G 6
                [-1, -1, -1]  # H 7
            ], dtype=np.float32)

        self.indices = np.array([
                2, 3, 6, 7, 5, 3, 1, 2, 0, 6, 4, 5, 0, 1
            ],  dtype=np.uint32)

        triangles = []
        for i in range(len(self.indices) - 2):
            triangles += [[self.indices[i], self.indices[i + 1], self.indices[i + 2]]]
        
        def surfaceNormal(A, B, C):
            AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
            AC = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
            n = np.cross(AB, AC)
            return n
        
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: 
                return v
            return v / norm
        
        vertexNormals = np.zeros((len(self.vertices), 3))
        
        for i in triangles:
            surfaceNormals = surfaceNormal(self.vertices[i[0]], self.vertices[i[1]], self.vertices[i[2]])
            vertexNormals[i[0]] += surfaceNormals
            vertexNormals[i[1]] += surfaceNormals
            vertexNormals[i[2]] += surfaceNormals
        
        vertexNormals = list(map(lambda x : normalize(x), vertexNormals))

        self.normals = np.array(vertexNormals, dtype=np.float32)

        # colors: RGB format
        self.colors = np.array([
                [1.0, 1.0, 1.0], 
                [1.0, 1.0, 0.0], 
                [1.0, 0.0, 1.0], 
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0], 
                [0.0, 1.0, 0.0], 
                [0.0, 0.0, 1.0], 
                [0.0, 0.0, 0.0]
            ], dtype=np.float32)

        
        
        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #
     

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, stride=0, offset=None)
        self.vao.add_ebo(self.indices)

        normalMat = np.identity(4, 'f')
        # projection = T.ortho(-1, 1, -1, 1, -1, 1)
        # modelview = np.identity(4, 'f')

        # Light
        I_light = np.array([
            [0.9, 0.4, 0.6],  # diffuse
            [0.9, 0.4, 0.6],  # specular
            [0.9, 0.4, 0.6]  # ambient
        ], dtype=np.float32)
        light_pos = np.array([0, 0.5, 0.9], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.6, 0.4, 0.7],  # diffuse
            [0.6, 0.4, 0.7],  # specular
            [0.6, 0.4, 0.7]  # ambient
        ], dtype=np.float32)

        shininess = 100.0
        phong_factor = 0.2
        mode = 1

        GL.glUseProgram(self.shader.render_idx)
        
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        # self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        # self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        
        return self

    def draw(self, projection, view, model):
        modelview = view
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

