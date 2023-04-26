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
    vertices, indices, color, triangles, texcoords = [], [], [], [], []
    
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
    for i in vertices:
        color += [[0, 1, 0]]
    
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
    
    # Calculating texture coordinates
    for i in range(stk + 1):
        for j in range(sec + 1):
            x = j / (int(np.sqrt(len(xList))) - 1)
            y = i / (int(np.sqrt(len(zList))) - 1)
            texcoords += [[x , y]]
    
    texcoords = np.array(texcoords, dtype=np.float32)
    
    return vertices, indices, color, normals, texcoords

class MeshSGD(object):
    def __init__(self, vert_shader, frag_shader):
        self.earthVertices, self.earthIndices, self.earthColors, self.earthNormals, self.earthTexcoords = sphere([0.0, 0.0, 10.0], 2.0, 50, 50) # center, radius, stacks, sectors - Sphere with stacks and sectors
        
        self.moonVertices, self.moonIndices, self.moonColors, self.moonNormals, self.moonTexcoords = sphere([0.0, 0.0, 13.0], 1.0, 50, 50) # center, radius, stacks, sectors - Sphere with stacks and sectors
        
        self.sunVertices, self.sunIndices, self.sunColors, self.sunNormals, self.sunTexcoords = sphere([0.0, 0.0, 0.0], 6.0, 50, 50) # center, radius, stacks, sectors - Sphere with stacks and sectors
        
        self.vao = VAO()
        self.uma1 = UManager(self.shader)
        self.vao1 = VAO()
        self.uma2 = UManager(self.shader)
        self.vao2 = VAO()
        self.uma3 = UManager(self.shader)
        
        self.shader = Shader(vert_shader, frag_shader)
        
    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # Earth
        self.vao.add_vbo(0, self.earthVertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.earthColors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.earthNormals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(3, self.earthTexcoords, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.earthIndices)

        # Moon
        self.vao1.add_vbo(0, self.moonVertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao1.add_vbo(1, self.moonColors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao1.add_vbo(2, self.moonNormals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao1.add_vbo(3, self.moonTexcoords, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao1.add_ebo(self.sunIndices)

        # Sun
        self.vao2.add_vbo(0, self.sunVertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao2.add_vbo(1, self.sunColors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao2.add_vbo(2, self.sunNormals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao2.add_vbo(3, self.sunTexcoords, ncomponents=2, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao2.add_ebo(self.earthIndices)

        normalMat = np.identity(4, 'f')
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')

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
        mode = 1

        GL.glUseProgram(self.shader.render_idx)
        
        # Earth
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        
        self.uma.setup_texture("texture", "./image/earth.jpg")
        
        # Moon
        self.uma1.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma1.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma1.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma1.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma1.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma1.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma1.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma1.upload_uniform_scalar1i(mode, 'mode')
        
        self.uma1.setup_texture("texture", "./image/earth.jpg")
        
        # Sun
        self.uma2.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma2.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma2.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma2.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma2.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma2.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma2.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma2.upload_uniform_scalar1i(mode, 'mode')
        
        self.uma2.setup_texture("texture", "./image/earth.jpg")
        
        return self

    def draw(self, projection, modelview, model):
        GL.glUseProgram(self.shader.render_idx)
        # Earth
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.earthIndices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        # Moon
        self.uma1.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma1.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.vao1.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.moonIndices.shape[0], GL.GL_UNSIGNED_INT, None)

        # Sun
        self.uma2.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma2.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.vao2.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.sunIndices.shape[0], GL.GL_UNSIGNED_INT, None)
        
    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

