import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def ellipsoid(rx, ry, rz, stk, sec):
    vertices, indices, color, triangles = [], [], [], []
    
    # Calculating vertex list
    stackMesh, sectorMesh = np.meshgrid(np.arange(0, stk + 1, 1), np.arange(0, sec + 1, 1))

    phiMesh = np.pi / 2 - np.pi * stackMesh / stk
    thetaMesh = 2 * np.pi * sectorMesh / sec 
    
    xMesh = rx * np.cos(phiMesh) * np.cos(thetaMesh)
    yMesh = ry * np.cos(phiMesh) * np.sin(thetaMesh)
    zMesh = rz * np.sin(phiMesh)
    
    xList = xMesh.flatten(order='F')
    yList = yMesh.flatten(order='F')
    zList = zMesh.flatten(order='F')
    
    vertices = list(map(lambda x, y, z: [x, y, z], xList, yList, zList))
    
    vertices = np.array(vertices, dtype=np.float32)

    # Calculating index list
    for i in range(stk):
        k1 = i * (sec + 1)
        k2 = k1 + sec + 1
        for j in range(sec):
            if i != 0:
                indices += [k1, k2, k1 + 1]
                triangles += [[k1, k2, k1 + 1]]
            if i != (stk - 1):
                indices += [k1 + 1, k2, k2 + 1]
                triangles += [[k1 + 1, k2, k2 + 1]]
            k1 += 1
            k2 += 1

    indices = np.array(indices, dtype=np.uint32)

    # Calculating vertex color
    for i in range(stk + 1):
        for j in range(sec + 1):
            if i % 2 == 0 and j % 2 == 0:
                color += [1, 0, 0]
            elif i % 2 == 0 and j % 2 != 0:
                color += [0, 0, 1]
            elif i % 2 != 0 and j % 2 == 0:
                color += [0, 0, 1]
            elif i % 2 != 0 and j % 2 != 0:
                color += [1, 0, 0]

    color = np.array(color, dtype=np.float32)

    # Calculating vertex normals
    def surfaceNormal(A, B, C):
        AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
        AC = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
        res = np.cross(AB, AC)
        return res

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    vertexNormals = np.zeros((len(vertices), 3))
    
    for i in triangles:
        surfaceNormals = surfaceNormal(vertices[i[0]], vertices[i[1]], vertices[i[2]])
        vertexNormals[i[0]] += surfaceNormals
        vertexNormals[i[1]] += surfaceNormals
        vertexNormals[i[2]] += surfaceNormals
    
    vertexNormals = list(map(lambda x : normalize(x), vertexNormals))
    
    normals = np.array(vertexNormals, dtype=np.float32)
    
    return vertices, indices, color, normals

class Ellipsoid(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors, self.normals = ellipsoid(1, 1.5, 2, 100, 100) # xRadius, yRadius, zRadius, stacks, sectors
        
        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

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

