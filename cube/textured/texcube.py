import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

class TexCube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices = np.array([
                [+1, +1, +1], # A 0
                [+1, +1, -1], # B 1
                [+1, -1, +1], # C 2
                [+1, -1, -1], # D 3
                [-1, +1, +1], # E 4
                [-1, +1, -1], # F 5
                [-1, -1, +1], # G 6
                [-1, -1, -1], # H 7
                    
                [+1, +1, +1], # A 8
                [+1, +1, -1], # B 9
                [+1, -1, +1], # C 10
                [+1, -1, -1], # D 11
                [-1, +1, +1], # E 12
                [-1, +1, -1], # F 13
                [-1, -1, +1], # G 14
                [-1, -1, -1], # H 15
                
                [+1, +1, +1], # A 16
                [+1, +1, -1], # B 17
                [+1, -1, +1], # C 18
                [+1, -1, -1], # D 19
                [-1, +1, +1], # E 20
                [-1, +1, -1], # F 21
                [-1, -1, +1], # G 22
                [-1, -1, -1]  # H 23
            ], dtype=np.float32)

        self.indices = np.array([
                0, 1, 2,
                1, 2, 3,
                4, 5, 6,
                5, 6, 7,
                
                12, 8, 14,
                8, 14, 10,
                9, 13, 11,
                13, 11, 15,
                
                21, 17, 20,
                17, 20, 16,
                22, 18, 23,
                18, 23, 19 
            ],  dtype=np.uint32)

        # colors: RGB format
        self.colors = np.array([
                [1.0, 1.0, 1.0], 
                [1.0, 1.0, 0.0], 
                [1.0, 0.0, 1.0], 
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0], 
                [0.0, 1.0, 0.0], 
                [0.0, 0.0, 1.0], 
                [0.0, 0.0, 0.0],
                
                [1.0, 1.0, 1.0], 
                [1.0, 1.0, 0.0], 
                [1.0, 0.0, 1.0], 
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0], 
                [0.0, 1.0, 0.0], 
                [0.0, 0.0, 1.0], 
                [0.0, 0.0, 0.0],
                
                [1.0, 1.0, 1.0], 
                [1.0, 1.0, 0.0], 
                [1.0, 0.0, 1.0], 
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0], 
                [0.0, 1.0, 0.0], 
                [0.0, 0.0, 1.0], 
                [0.0, 0.0, 0.0]
            ], dtype=np.float32)

        self.texcoords = np.array([
                [0.0, 1/3], # A 0 
                [1/4, 1/3], # B 1 
                [0.0, 2/3], # C 2
                [1/4, 2/3], # D 3
                [3/4, 1/3], # E 4
                [2/4, 1/3], # F 5
                [3/4, 2/3], # G 6
                [2/4, 2/3], # H 7
                
                [2/4, 1/3], # A 8
                [3/4, 1/3], # B 9
                [2/4, 2/3], # C 10
                [3/4, 2/3], # D 11
                [1/4, 1/3], # E 12
                [1.0, 1/3], # F 13
                [1/4, 2/3], # G 14
                [1.0, 2/3], # H 15
                
                [3/4, 1/3], # A 16
                [3/4, 0.0], # B 17
                [3/4, 2/3], # C 18
                [3/4, 1.0], # D 19
                [2/4, 1/3], # E 20
                [2/4, 0.0], # F 21
                [2/4, 2/3], # G 22
                [2/4, 1.0]  # H 23
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
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, dtype=GL.GL_FLOAT, stride=0, offset=None)
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
        
        self.uma.setup_texture("texture", "./textured/image/texture.jpeg")
        
        return self

    def draw(self, projection, view, model):
        modelview = view
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

