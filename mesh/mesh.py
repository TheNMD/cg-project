import sys
sys.path.append("../libs")

from shader import *
import transform as T
from buffer import *
import ctypes
import glfw

import OpenGL.GL as GL
import numpy as np

def mesh(xFirst, xLast, zFirst, zLast):
    
    def randFunc(x, y):
        res = np.sin(x) + np.cos(y)
        return res
    
    def surfaceNormal(A, B, C):
        AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
        AC = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
        BA = [A[0] - B[0], A[1] - B[1], A[2] - B[2]] 
        BC = [C[0] - B[0], C[1] - B[1], C[2] - B[2]]
        CA = [A[0] - C[0], A[1] - C[1], A[2] - C[2]]
        CB = [B[0] - C[0], B[1] - C[1], B[2] - C[2]]
        nA = np.cross(AB, AC)
        nB = np.cross(BA, BC)
        nC = np.cross(CA, CB)
        return [nA, nB, nC]
    
    vertices, indices, triangles, normals, color = [], [], [], [], []
    
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

    for i in range(len(vertices)):
        yColor = (vertices[i][1] + abs(yMin)) / (yMax + abs(yMin))
        color += [yColor, 0, 1 - yColor]
        # Red means y is higher
        # Blue means y is lower
    
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

    for i in triangles:
        normals += surfaceNormal(vertices[i[0]], vertices[i[1]], vertices[i[2]])
        break
    
    print(normals)
        
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    color = np.array(color, dtype=np.float32)
    
    return vertices, indices, color

class Mesh(object):
    def __init__(self, vert_shader, frag_shader):
        self.vertices, self.indices, self.colors = mesh(-20, 20, -20, 20) # xFirst, xLast, zFirst, zLast
        
        # self.normals = [] # YOUR CODE HERE to compute vertex's normal using the coordinates
        
        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        #
     

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

