"""
Transformations on trimesh.base.meshes
"""

from typing import Type, Callable
import trimesh
import numpy as np
import copy



def get_bbox_dimensions(mesh: Type[trimesh.base.Trimesh]) -> Type[np.ndarray]:
    """ Getsthe dimensions of the bounding box that contains the object
    """
    return np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)

def get_center_bbox(mesh: Type[trimesh.base.Trimesh]) -> Type[np.ndarray]:
    """ Returns the center of the bounding box of the vertices.
    """
    return 0.5 * (np.min(mesh.vertices, axis=0) + np.max(mesh.vertices, axis=0))

def get_com_vertices(mesh: Type[trimesh.base.Trimesh]) -> Type[np.ndarray]:
    """ Returns the center of mass of vertices
    """
    return np.mean(mesh.vertices, axis=0)

def recenter(mesh: Type[trimesh.base.Trimesh],
             center_fn: Callable[[Type[trimesh.base.Trimesh]], Type[np.ndarray]],
             in_place:bool = True) -> Type[trimesh.base.Trimesh]:
    """ Returns a recentered copy of the input mesh
    """ 
    center = center_fn(mesh) 
    mesh_ = mesh if in_place else copy.deepcopy(mesh)
    mesh_.vertices = mesh.vertices - center
    mesh = mesh_
    return mesh

def global_scale(mesh: Type[trimesh.base.Trimesh],
                scale: float,
                in_place: bool = True) -> Type[trimesh.base.Trimesh]:
    mesh_ = mesh if in_place else copy.deepcopy(mesh)
    mesh_.vertices *= scale
    mesh = mesh_
    return mesh

def global_rotation(mesh: Type[trimesh.base.Trimesh],
                    rotation_matrix: Type[np.ndarray],
                    in_place: bool = True) -> Type[trimesh.base.Trimesh]:
    """ Rotates the vertices around the origin
    """  
    mesh_ = mesh if in_place else copy.deepcopy(mesh)
    mesh_.vertices = np.dot(rotation_matrix, mesh_.vertices.transpose()).transpose()
    mesh = mesh_
    return mesh

def resize_to_unit_cube(
    mesh: Type[trimesh.base.Trimesh],
    in_place: bool = True) -> Type[trimesh.base.Trimesh]:
    """ Rescales the object such that its bounding box fits to unit cube
    """
    dimensions = get_bbox_dimensions(mesh)
    scale = np.min(1/dimensions) 
    mesh = global_scale(mesh, scale, in_place)
    return mesh 

if __name__=='__main__':

    from utils.convert import *
    glb = glbfile2glb('/orion/group/objaverse/hf-objaverse-v1/glbs/000-044/fff7ca82f0c64f429b45bf468525d35a.glb')
    mesh = glb2trimesh(glb) 
    
    print(f"dimensions before : {get_bbox_dimensions(mesh)}")
    print(f"dimensions after unit_cube: {get_bbox_dimensions(resize_to_unit_cube(mesh))}")
    print(f"bbox center before: {get_center_bbox(mesh)}") 
    print(f"bbox center after recenter: {get_center_bbox(recenter(mesh=mesh, center_fn=get_center_bbox ))}")
    
    pass