import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def sphere1(depth):
    vertices, indices, color, triangles, texcoords = [], [], [], [], []
    
    # Calculating vertex list and main tetrahedron texture coordinates
    vertices = [
                [0.0, 1.0, 0.0],             # O1  0
                [0.0, -1.0, np.sqrt(3)],     # A1  1
                [2.0, -1.0, -np.sqrt(3)],    # B1  2
                [0.0, 1.0, 0.0],             # O2  3
                [2.0, -1.0, -np.sqrt(3)],    # B2  4
                [-2.0, -1.0, -np.sqrt(3)],   # C2  5
                [0.0, 1.0, 0.0],             # O3  6
                [-2.0, -1.0, -np.sqrt(3)],   # C3  7
                [0.0, -1.0, np.sqrt(3)],     # A3  8
                [0.0, -1.0, np.sqrt(3)],     # A4  9
                [2.0, -1.0, -np.sqrt(3)],    # B4  10
                [-2.0, -1.0, -np.sqrt(3)],   # C4  11
               ]

    indices = [0, 1, 2] + [3, 4, 5] + [6, 7, 8] + [9, 10, 11]
    
    triangles += [[0, 1, 2]] + [[3, 4, 5]] + [[6, 7, 8]] + [[9, 10, 11]]

    texcoords = [
                [0.125, 0.0],    # O1 0
                [0.0, 1.0],      # A1 1
                [0.25, 1.0],     # B1 2
                [0.375, 0.0],    # O2 3
                [0.25, 1.0],     # B2 4
                [0.5, 1.0],      # C2 5
                [0.625, 0.0],    # O3 6
                [0.5, 1.0],      # A3 7
                [0.75, 1.0],     # C3 8
                [0.875, 0.0],    # A4 9
                [0.75, 1.0],     # B4 10
                [1.0, 1.0],      # C4 11
                ]
    
    # Calculating index list and sub-tetrahedrons texture coordinates
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    
    for i in range(len(vertices)):
        vertices[i] = normalize(vertices[i])
    
    for i in range(depth):
        indices_temp = []
        triangles_temp = []
        for j in triangles:
            A = vertices[j[0]]
            B = vertices[j[1]]
            C = vertices[j[2]]
            
            D = normalize((A + B) / 2)
            E = normalize((B + C) / 2)
            F = normalize((C + A) / 2)
            
            vertices += [D] +  [E] + [F]
            
            indexA, indexB, indexC = j[0], j[1], j[2]
            indexD, indexE, indexF = len(vertices) - 3, len(vertices) - 2, len(vertices) - 1
            
            indices_temp += [indexA, indexD, indexF] + [indexB, indexE, indexD] + [indexC, indexF, indexE] + [indexD, indexE, indexF]
            triangles_temp += [[indexA, indexD, indexF]] + [[indexB, indexE, indexD]] + [[indexC, indexF, indexE]] + [[indexD, indexE, indexF]]
            
            ATex = texcoords[j[0]]
            BTex = texcoords[j[1]]
            CTex = texcoords[j[2]]
            
            DTex = [(ATex[0] + BTex[0]) / 2, (ATex[1] + BTex[1]) / 2]
            ETex = [(BTex[0] + CTex[0]) / 2, (BTex[1] + CTex[1]) / 2]
            FTex = [(CTex[0] + ATex[0]) / 2, (CTex[1] + ATex[1]) / 2]
            
            texcoords += [DTex] + [ETex] + [FTex]
        indices = indices_temp
        triangles = triangles_temp

    vertices = np.array(vertices, dtype=np.float32)
    
    indices = np.array(indices, dtype=np.uint32)
    
    texcoords = np.array(texcoords, dtype=np.float32)
    
    # Calculating vertex color
    for i in vertices:
        color += [1, 1, 1]

    color = np.array(color, dtype=np.float32)
    
    # Calculating vertex normals
    def surfaceNormal(A, B, C):
        AB = B - A
        AC = C - A
        res = np.cross(AB, AC)
        return res
    
    vertexNormals = np.zeros((len(vertices), 3))
    
    for i in triangles:
        surfaceNormals = surfaceNormal(vertices[i[0]], vertices[i[1]], vertices[i[2]])
        vertexNormals[i[0]] += surfaceNormals
        vertexNormals[i[1]] += surfaceNormals
        vertexNormals[i[2]] += surfaceNormals
    
    vertexNormals = list(map(lambda x : normalize(x), vertexNormals))
    
    normals = np.array(vertexNormals, dtype=np.float32)
    
    texcoords = np.array(texcoords, dtype=np.float32)

    return vertices, indices, color, normals, texcoords

class TexSphere1(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors, self.normals, self.texcoords = sphere1(6) # subdivision - Sphere from tetrahedron
        
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

        GL.glUseProgram(self.shader.render_idx)
        
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        # self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        # self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')
        
        self.uma.setup_texture("texture", "./textured/image/texture.png")
        
        return self

    def draw(self, projection, modelview, model):
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

