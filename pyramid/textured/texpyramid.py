import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

class TexPyramid(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
                [0.0, 1.0, 0.0],    # O 0
                [-1.5, -1.0, -1.5], # D 1
                [1.5, -1.0, -1.5],  # C 2
                [1.5, -1.0, 1.5],   # B 3
                [-1.5, -1.0, 1.5]   # A 4
            ], dtype = np.float32)

        self.indices = np.array([
                1, 0, 2, 3, 1, 4, 0 ,3
            ], dtype = np.uint32)

        triangles = []
        for i in range(len(self.indices) - 2):
            triangles += [[self.indices[i], self.indices[i + 1], self.indices[i + 2]]]
        
        def surfaceNormal(A, B, C):
            AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
            AC = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
            n = np.cross(AB, AC)
            return n
        
        vertexNormals = np.zeros((len(self.vertices), 3))
        for i in triangles:
            surfaceNormals = surfaceNormal(self.vertices[i[0]], self.vertices[i[1]], self.vertices[i[2]])
            vertexNormals[i[0]] += surfaceNormals
            vertexNormals[i[1]] += surfaceNormals
            vertexNormals[i[2]] += surfaceNormals
        
        for i in self.vertices:
            i = i / np.linalg.norm(i)

        self.normals = np.array(vertexNormals, dtype=np.float32)

        # colors: RGB format
        self.colors = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0], 
                [0.0, 0.0, 1.0], 
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0], 
            ],dtype= np.float32)

        self.texcoord = np.array([
                [0.5, 0.5],    # O 0
                [0.0, 1.0],    # D 1
                [1.0, 1.0],    # C 2
                [1.0, 0.0],    # B 3
                [0.0, 0.0]     # A 4
            ], dtype = np.float32)
        
        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        
        self.selected_texture = 1

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoord, ncomponents=2, dtype=GL.GL_FLOAT, stride=0, offset=None)
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

        GL.glUseProgram(self.shader.render_idx)
        
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        # self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        # self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')
        
        self.uma.setup_texture("texture", "./textured/image/test.png")
        # self.uma.setup_texture("texture", "./textured/image/test1.png")
        
        return self

    def draw(self, projection, modelview, model):
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

