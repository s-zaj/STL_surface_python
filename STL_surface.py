from __future__ import division


import numpy as np

import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.tri as mtri

import scipy
from scipy.spatial import Delaunay
from scipy.spatial import voronoi_plot_2d,Voronoi

from mpl_toolkits.mplot3d.axes3d import Axes3D

from operator import itemgetter

import random

import struct

# def find_nearest_value(img,val):
#     value = np.min(np.abs(img-val)) 
#     return value

# def SamplePoints(img,val,sensitivity,density):
#   points = np.abs(img-val) < sensitivity #find a bunch of bands
#   indices = np.where(points)
#   # randIndices = random.sample(range(len(indices[0])), density)
#   randSample = [[indices[0][x],indices[1][x]] for x in sorted(random.sample(xrange(len(indices[0])), density)) ]
    
#   return randSample

class StlWriter(object):
   
    def __init__(self, f):
        self._f = f
    
    @classmethod
    def write(cls, fname, mesh, name='', bin=True):
        """ write(fname, mesh, name='', bin=True)
       
        This classmethod is the entry point for writing mesh data to STL.
       
        Parameters
        ----------
        fname : string
            The filename to write to.
        mesh : Mesh data
            Can be np.ndarray.
        name : string
            The name of the object (e.g. 'teapot')
        bin : bool
            Whether to write binary, which is much more compact then ascii.
       
        """
        if isinstance(mesh, np.ndarray):
            if mesh.shape[1] != 3:
                raise ValueError('Mesh vertices must be 3D.')
            vertex1 = mesh[:,0,:] #vertex1
            vertex2 = mesh[:,1,:] #counterclock-wise vertex2
            vertex3 = mesh[:,2,:] #counterclock-wise vertex3
        else:
            raise ValueError('Unknown type for mesh vertices.')
       
        # Open file
        f = open(fname, 'wb')
        try:
            # Get writer and write head
            if bin:
                writer = StlBinWriter(f)
                f.write(struct.pack('<B', 0)*80) #HEADER, NO SIGNIFICANCE
                f.write(struct.pack('<I', len(vertex1)))
            # Write vertices
            for i in range(len(vertex1)):
                writer.writeFace(vertex1[i], vertex2[i], vertex3[i])        
            # Write end
            if not bin:
                writer.writeLine('endsolid %s' % name)
        except EOFError:
            pass
        finally:
            f.close()

class StlBinWriter(StlWriter):
   
    def writeFace(self, v1, v2, v3):
        """ writeFace(v1, v2, v3)
       
        Write the three vertices that make up a face.
       
        """
        # Construct data
        dataList = []
        N = np.empty(3) # Unit Normal Vector, normalized crossproduct of two triangle edges
        V = v2 - v1
        W = v3 - v1
        N[0] = (V[1] * W[2]) - (V[2]*W[1]) # Nx
        N[1] = (V[2] * W[0]) - (V[0]*W[2]) # Ny
        N[2] = (V[0] * W[1]) - (V[1]*W[0]) # Nz
        magnitude = (abs(N[0]) + abs(N[1]) + abs(N[2]))
        N /= magnitude # Normalize to get unit normal

        for p in N:
            dataList.append( struct.pack('<f', p) ) #SURFACE NORMAL, REDUNDANT (can be 0)
        for p in v1:
            dataList.append( struct.pack('<f', p) ) # VERTEX 1
        for p in v2:
            dataList.append( struct.pack('<f', p) ) # VERTEX 2
        for p in v3:
            dataList.append( struct.pack('<f', p) ) # VERTEX 3
        if True:
            dataList.append( struct.pack('<H', 0) ) #"ATTRIBUTE", UNSIGNED INT 2 BYTES
       
        # Write data
        data = ''.encode('ascii').join(dataList)
        self._f.write(data)


def GetContourPoints(img,N,scale):
    # This gets the vertices of the paths that make up the contour plot. 
    # N is the number of contours
    paths = list()
    levels = list()
    xAxis = xrange(np.shape(img)[0])
    yAxis = xrange(np.shape(img)[1])
    contours = plt.contour(xAxis,yAxis,img,N)
    plt.clf() #clear the figure created above
    for i,path in enumerate(contours.collections):
        level = contours.levels[i]
        for j,newPath in enumerate(path.get_paths()):
            vertices = newPath.vertices
            levelVertices = np.ones(np.shape(vertices)[0])*level*scale
            paths.append([vertices[:,0],vertices[:,1],levelVertices])
    
    return paths

# def GetContourPoints(img,N):
#   # This gets the vertices of the paths that make up the contour plot. 
#   # N is the number of contours
#   paths = list()
#   xAxis = xrange(np.shape(img)[0])
#   yAxis = xrange(np.shape(img)[1])
#   contours = plt.contour(xAxis,yAxis,img,N)
#   # plt.clf() #clear the figure created above
#   for i,path in enumerate(contours.collections):
#       newPath = path.get_paths()[1]
#       vertices = newPath.vertices
#       paths.append([vertices[:,0],vertices[:,1]])

#   return paths

def SampleContours(paths,density,type='uniform'):
    # Uniform or random sampling. Probably want to mix the two a little bit?
    # !!! density parameter doesn't behave the same way for uniform and random sampling!!
    sampledPaths = list()
    for iPath,path in enumerate(paths):
        length = len(path[0])
        
        if type =='uniform':
            indices = xrange(0,length,density)
        if type =='random':
            if density > length:
                while density > length:
                    density -= 1
            indices = sorted(random.sample(xrange(length), density))
        
        xPoints = [path[0][i] for i in indices]
        yPoints = [path[1][i] for i in indices]
        zPoints = [path[2][i] for i in indices]
        sampledPaths.append([xPoints,yPoints,zPoints])
    return sampledPaths


def TwoDimensionalSine(center_x,center_y,sizex,sizey,freq,decay):
    # Create a 2d image with a sine wave
    sin = []
    env = []
    x = np.linspace(-center_x/sizex, (sizex-center_x)/sizex, sizex)
    y = np.linspace(-center_y/sizey, (sizey-center_y)/sizey, sizey)

    xx,yy = np.meshgrid(x,y)

    sin = np.cos((xx**2 + yy**2) * freq)
    env = np.exp(-(xx**2 + yy**2) * decay) 

    return sin, env

#-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
## CREATE THE TEST IMAGE
sizex = 2000
sizey = 2000
scale = 100

freq1 = 30
decay1 = 6
z1_sin, z1_env = TwoDimensionalSine(100,1200,sizex,sizey,freq1,decay1)
z1 = z1_sin * z1_env

freq2 = 45
decay2 = 13
z2_sin, z2_env = TwoDimensionalSine(1600,1050,sizex,sizey,freq2,decay2)
z2 = z2_sin * z2_env


freq3 = 41
decay3 = 4
z3_sin, z3_env = TwoDimensionalSine(1800,1000,sizex,sizey,freq3,decay3)
z3 = z3_sin * z3_env
#-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:

contours = GetContourPoints(z1+z2+z3,300,scale)
sampledContours = SampleContours(contours,10,type='uniform')

# Plots the sampled contours
# for i,contour in enumerate(sampledContours):
#     # plt.plot(contour[0], contour[1], linewidth=0.5, color='r')
#     plt.plot(contour[0], contour[1], 'o')
# plt.imshow(z1 + z2 + z3, cmap=cm.gist_earth)
# plt.show()

allPoints = []
for i,contour in enumerate(sampledContours):
    allPoints.extend([list(a) for a in zip(contour[0],contour[1],contour[2])])

allPoints = np.asarray(allPoints)
tri = mtri.Triangulation(allPoints[:,0], allPoints[:,1])
mesh = allPoints[tri.triangles]
del allPoints # Free some memory since tri.x & tri.y have the data now

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_trisurf(allPoints[:,0], allPoints[:,1], allPoints[:,2], triangles=tri.triangles, cmap=plt.cm.Spectral)

# plt.show()

d = StlWriter.write('abcd.stl',mesh,'teapot', bin=True)





