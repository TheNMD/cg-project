import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def sphere(center, r, stk, sec):   
    vertices, indices, color, triangles = [], [], [], []
    
    # Calculating vertex list
    stackMesh, sectorMesh = np.meshgrid(np.arange(0, stk + 1, 1), np.arange(0, sec + 1, 1))

    phiMesh = np.pi / 2 - np.pi * stackMesh / stk
    thetaMesh = 2 * np.pi * sectorMesh / sec 
    
    xMesh = center[0] + r * np.cos(phiMesh) * np.cos(thetaMesh)
    yMesh = center[1] + r * np.cos(phiMesh) * np.sin(thetaMesh)
    zMesh = center[2] + r * np.sin(phiMesh)
    
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

def sphere1(depth):
    vertices, indices, color, triangles = [], [], [], []
    
    # Calculating vertex list
    vertices = [[0.0, 1.0, 0.0],
                [0.0, -0.5, 0.8165],
                [0.7071, -0.5, -0.4082],
                [-0.7071, -0.5, -0.4082]]

    indices = [0, 1, 2] + [0, 2, 3] + [0, 3, 1] + [1, 3, 2]
    
    triangles += [[0, 1, 2]] + [[0, 2, 3]] + [[0, 3, 1]] + [[1, 3, 2]]

    # Calculating index list
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
            
            D = normalize([(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2])
            E = normalize([(B[0] + C[0]) / 2, (B[1] + C[1]) / 2, (B[2] + C[2]) / 2])
            F = normalize([(C[0] + A[0]) / 2, (C[1] + A[1]) / 2, (C[2] + A[2]) / 2])
            
            vertices += [D] +  [E] + [F]
            
            indexA, indexB, indexC = j[0], j[1], j[2]
            indexD, indexE, indexF = len(vertices) - 3, len(vertices) - 2, len(vertices) - 1
            
            indices_temp += [indexA, indexD, indexF] + [indexB, indexE, indexD] + [indexC, indexF, indexE] + [indexD, indexE, indexF]
            triangles_temp += [[indexA, indexD, indexF]] + [[indexB, indexE, indexD]] + [[indexC, indexF, indexE]] + [[indexD, indexE, indexF]]
        indices = indices_temp
        triangles = triangles_temp

    vertices = np.array(vertices, dtype=np.float32)
    
    indices = np.array(indices, dtype=np.uint32)

    # Calculating vertex color
    for i in vertices:
        color += [1, 0, 0]

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

    return vertices, indices, color, normals

class Sphere(object):
    def __init__(self, vert_shader, frag_shader, flag):
        self.flag = flag
        if self.flag == 0:
            self.vertices, self.indices, self.colors, self.normals = sphere([0.0, 0.0, 0.0], 1, 50, 50) # center, radius, stacks, sectors - Sphere with stacks and sectors
        else:
            self.vertices, self.indices, self.colors, self.normals = sphere1(5) # subdivision - Sphere from tetrahedron
        
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
        # No need EBO for GL.glDrawArray(GL.GL_TRIANGLES, 0, self.indices.shape[0])
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
        if self.flag == 0:
            GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
            # GL.glDrawArray(GL.GL_TRIANGLES, 0, self.indices.shape[0])

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

