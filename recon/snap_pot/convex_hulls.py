"""Double-precision Qhull helper called by Blender over JSON stdin/stdout.

Uses external Python with NumPy/SciPy because Blender ships float32 BMesh hulls.
No coordinates are rescaled or perturbed; only hull faces are canonicalized.
"""
import json
import sys

import numpy as np
from scipy.spatial import ConvexHull


def hull(points):
    vertices=np.asarray(points,dtype=np.float64)
    h=ConvexHull(vertices)
    faces=h.simplices.copy()
    triangles=vertices[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    reverse=np.einsum('ij,ij->i',normals,h.equations[:,:3])<0
    faces[reverse]=faces[reverse][:,::-1]
    return faces.tolist()


if __name__=='__main__':
    json.dump([hull(v) for v in json.load(sys.stdin)],sys.stdout)
