import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def mesh(xFirst, xLast, zFirst, zLast, step):
    vertices, indices, color, triangles = [], [], [], []
    
    # Calculating vertex list
    xMesh, zMesh = np.meshgrid(np.arange(xFirst, xLast + (xLast - xFirst) / step, (xLast - xFirst) / step), np.arange(zFirst, zLast + (zLast - zFirst) / step, (zLast - zFirst) / step))
    yMesh = xMesh**2 + zMesh**2
    # yMesh = np.sin(xMesh) + np.cos(zMesh)
    yMax, yMin = yMesh.max(), yMesh.min()
    
    xList = xMesh.flatten()
    yList = yMesh.flatten()
    zList = zMesh.flatten()
        
    vertices = list(map(lambda x, y, z: [x, y, z], xList, yList, zList))
    
    vertices = np.array(vertices, dtype=np.float32)

    # Calculating index list
    s1 = int(np.sqrt(len(xList))) - 1
    s2 = int(np.sqrt(len(zList))) - 1
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
        if i != s1 - 1:
            k1 = k2
            k2 = k1 + s2 + 1
            for j in range(s2):
                indices += [k1, k2, k2 - 1] +  [k2 - 1, k1, k1 - 1]
                triangles += [[k1, k2, k2 - 1]] + [[k2 - 1, k1, k1 - 1]]
                if (j == s2 - 1):
                    indices += [k2 - 1, k2 - 1]
                k1 -= 1
                k2 -= 1

    indices = np.array(indices, dtype=np.uint32)

    # Calculating vertex color
    if yMax != yMin:
        yColor = list(map(lambda x : (x + abs(yMin)) / (yMax + abs(yMin)), vertices[:, 1]))
        color = list(map(lambda x : [x, 0, 1 - x], yColor))     
        # Red means y is higher
        # Blue means y is lower
    else:
        color = list((map(lambda x : 0 * x + [0, 1, 0], vertices)))

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

def SGD(initPoint, learningRate, iteration, vertices):
    maxIdx = np.argmax(vertices, axis=0)[1]
    minIdx = np.argmin(vertices, axis=0)[1]
    yMin = vertices[minIdx][1]

    num = initPoint # np.random.randint(0, len(vertices))
    xRes, yRes, zRes = vertices[num][0], vertices[num][1], vertices[num][2]
    counter = 0
    pathVertices = [[vertices[num][0], vertices[num][1], vertices[num][2]]]
    pathIndices = [counter]
    pathColors = [[0, 1, 0]]
    for i in range(iteration):
        num = np.random.randint(0, len(vertices))
        x = vertices[num][0]
        z = vertices[num][2]
        
        y = x**2 + z**2
        # y = np.sin(x) + np.cos(z)
        
        if y <= yRes:
            yRes = y
            
            xRes += x * 2 * learningRate
            zRes += z * 2 * learningRate
            # xRes += np.cos(x) * learningRate
            # zRes += -np.sin(z) * learningRate
            
            print(f"Iter {i}: x = {xRes}, y = {yRes}, z = {zRes}")
            
            counter += 1
            
            pathVertices += [[vertices[num][0], vertices[num][1], vertices[num][2]]]
            pathIndices += [counter]
            pathColors += [[0, 1, 1]]
            
    pathVertices = np.array(pathVertices, dtype=np.float32)
    pathIndices = np.array(pathIndices, dtype=np.uint32)
    pathColors = np.array(pathColors, dtype=np.float32)
    
    print(f"yMin = {yMin}")
    
    return pathVertices, pathIndices, pathColors

class MeshSGD(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors, self.normals = mesh(-10, 10, -10, 10, 200) # xFirst, xLast, zFirst, zLast, step
        
        self.pathVertices, self.pathIndices, self.pathColors = SGD(0, 0.001, 10000, self.vertices)
        
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        
        self.vao1 = VAO()
        self.shader1 = Shader(vert_shader, frag_shader)
        self.uma1 = UManager(self.shader1)
        
    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
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
        mode = 1

        GL.glUseProgram(self.shader.render_idx)
        
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        # self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        # self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        
        ##################################################
        
        self.vao1.add_vbo(0, self.pathVertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao1.add_vbo(1, self.pathColors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao1.add_ebo(self.pathIndices)
        
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
        mode = 1

        GL.glUseProgram(self.shader1.render_idx)
        self.uma1.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        # self.uma1.upload_uniform_matrix4fv(projection, 'projection', True)
        # self.uma1.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma1.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma1.upload_uniform_vector3fv(light_pos, 'light_pos')
        
        self.uma1.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        
        self.uma1.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma1.upload_uniform_scalar1i(mode, 'mode')
        
        return self

    def draw(self, projection, modelview, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao1.activate()
        GL.glUseProgram(self.shader1.render_idx)
        self.uma1.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma1.upload_uniform_matrix4fv(modelview, 'modelview', True)
        GL.glDrawElements(GL.GL_LINE_STRIP, self.pathIndices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2
