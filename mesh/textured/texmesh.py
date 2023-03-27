import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def texmesh(xFirst, xLast, zFirst, zLast):
    vertices, indices, color, triangles, texcoord = [], [], [], [], []
    
    def randFunc(x, y):
        # if x > 8 or x < -8 or y > 8 or y < -8:
        #     return 0
        res = np.sin(x) + np.cos(y)
        return res
    
    xList = np.arange(xFirst, xLast + (xLast - xFirst) / 100, (xLast - xFirst) / 100)
    zList = np.arange(zFirst, zLast + (zLast - zFirst) / 100, (zLast - zFirst) / 100)
    yMax, yMin = randFunc(xList[0], zList[0]), randFunc(xList[-1], zList[-1])
    
    for i in range(len(xList)):
        for j in range(len(zList)):
            x = xList[i]
            z = zList[j]
            y = randFunc(x, z)
            vertices += [[x, y, z]]
            if (y > yMax):
                yMax = y
            if (y < yMin):
                yMin = y
            if i % 2 == 0 and j % 2 == 0:
                texcoord += [[0.0 , 1.0]]
            elif i % 2 == 0 and j % 2 != 0:
                texcoord += [[0.0 , 0.0]]
            elif i % 2 != 0 and j % 2 == 0:
                texcoord += [[1.0 , 1.0]]
            elif i % 2 != 0 and j % 2 != 0:
                texcoord += [[1.0 , 0.0]]
    
    s1 = len(xList) - 1
    s2 = len(zList) - 1
    for i in range(0, s1, 2):
        k1 = i * (s2 + 1)
        k2 = k1 + s2 + 1
        for j in range(s2):
            indices += [k1, k1 + 1, k2] +  [k2, k2 + 1, k1 + 1]
            triangles += [[k1, k1 + 1, k2]] + [[k2, k2 + 1, k1 + 1]]
            if (j == s2 - 1):
                 indices += [k2 + 1, k2 + 1]
            k1 += 1
            k2 += 1
        if (i != s1 - 1):
            k1 = k2
            k2 = k1 + s2 + 1
            for j in range(s2):
                indices += [k1, k2, k2 - 1] +  [k2 - 1, k1, k1 - 1]
                triangles += [[k1, k2, k2 - 1]] + [[k2 - 1, k1, k1 - 1]]
                if (j == s2 - 1):
                    indices += [k2 - 1, k2 - 1]
                k1 -= 1
                k2 -= 1

    if yMax != yMin:
        for i in range(len(vertices)):
            yColor = (vertices[i][1] + abs(yMin)) / (yMax + abs(yMin))
            color += [yColor, 0, 1 - yColor]
            # Red means y is higher
            # Blue means y is lower
    else:
        for i in range(len(vertices)):
            color += [0, 0, 1]

    texcoord += [[0.5, 0.5]]

    def surfaceNormal(A, B, C):
        AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
        AC = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
        n = np.cross(AB, AC)
        return n

    vertexNormals = np.empty((len(vertices), 3))
    for i in triangles:
        surfaceNormals = surfaceNormal(vertices[i[0]], vertices[i[1]], vertices[i[2]])
        vertexNormals[i[0]] = surfaceNormals / np.linalg.norm(surfaceNormals)
        vertexNormals[i[1]] = surfaceNormals / np.linalg.norm(surfaceNormals)
        vertexNormals[i[2]] = surfaceNormals / np.linalg.norm(surfaceNormals)
    
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    color = np.array(color, dtype=np.float32)
    normals = np.array(vertexNormals, dtype=np.float32)
    texcoord = np.array(texcoord, dtype=np.float32)
    
    return vertices, indices, color, normals, texcoord

class TexMesh(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors, self.normals, self.texcoord = texmesh(-10, 10, -10, 10) # xFirst, xLast, zFirst, zLast
        
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
        
        self.uma.upload_uniform_scalar1i(self.selected_texture, 'selected_texture')
        self.uma.upload_uniform_scalar1i(1, 'face')
        
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

