import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np

def frustum(r, h1, h2, sides):
    vertices, indices, color, triangles = [], [], [], []
    
    # Calculating vertex list and vertex color
    ratio = h2 / h1

    sideList = np.arange(0, sides, 1)
    
    thetaList = 2 * np.pi * sideList / sides
    
    xList = r * np.cos(thetaList)
    zList = r * np.sin(thetaList)
    x1List = xList * ratio
    z1List = zList * ratio
    
    for i in range(len(thetaList)):
        vertices += [[xList[i], 1, zList[i]], [x1List[i], 1 + h1 * ratio, z1List[i]]]
        color += [0, 0, 1] + [1, 0, 0]

    # Calculating index list 
    # Sides
    for i in range(sides):
        k1 = i * 2
        k2 = k1 + 2
        if i != sides - 1:
            indices += [k1, k1 + 1, k2] +  [k1 + 1, k2 + 1, k2]
            triangles += [[k1, k1 + 1, k2]] + [[k1 + 1, k2 + 1, k2]]
        else:
            indices += [k1, k1 + 1, 0] +  [k1 + 1, 1, 0]
            triangles += [[k1, k1 + 1, 0]] + [[k1 + 1, 1, 0]]

    # Bottom
    vertices += [[0, 1, 0]]
    color += [0, 0, 1]
    for i in range(sides):
        k1 = i * 2
        k2 = k1 + 2
        if i != sides - 1:
            indices += [k1, len(vertices) - 1, k2]
            triangles += [[k1, len(vertices) - 1, k2]]
        else:
            indices += [k1, len(vertices) - 1, 0]
            triangles += [[k1, len(vertices) - 1, 0]]
    
    # Move from bottom to top
    indices += [0] + [1]
    
    # Top
    vertices += [[0,  1 + h1 * ratio, 0]]
    color += [1, 0, 0]
    for i in range(sides):
        k1 = i * 2 + 1
        k2 = k1 + 2
        if i != sides - 1:
            indices += [k1, len(vertices) - 1, k2]
            triangles += [[k1, len(vertices) - 1, k2]]
        else:
            indices += [k1, len(vertices) - 1, 1]
            triangles += [[k1, len(vertices) - 1, 1]]
    
    vertices = np.array(vertices, dtype=np.float32)
    
    indices = np.array(indices, dtype=np.uint32)
    
    color = np.array(color, dtype=np.float32)
    
    # Calculating vertex normals
    def surfaceNormal(A, B, C):
        AB = B - A
        AC = C - A
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

class Frustum(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors, self.normals = frustum(1, 2, 1, 4) # radius, height 1, height 2, sides
        
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

