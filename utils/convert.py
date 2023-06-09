
import os
import numpy as np
import trimesh
from utils.io import prepare_dir_of, file_location_type
from typing import Callable, Type


def objfile2trimesh(obj_path:str) -> Type[trimesh.base.Trimesh]:
    if file_location_type(obj_path) == 'local':
        mesh = trimesh.load(obj_path, force="mesh") # TODO LOADS SCENE??
        # with open(obj_path, 'r') as f:
        #     mesh = trimesh.base.Trimesh( **trimesh.exchange.obj.load_obj(f) )
    else:
        raise NotImplementedError(f"Uncaught loading instructions for .obj of type {file_location_type(obj_path)}")
    return mesh 

def glbfile2trimesh(glb_path: str) -> Type[trimesh.base.Trimesh]:
    return glb2trimesh(glbfile2glb(glb_path))

def glbfile2glb(glb_path:str) -> dict:
    if file_location_type(glb_path) == 'local':
        with open(glb_path, 'rb') as f:
            glb_dict = trimesh.exchange.gltf.load_glb(f)
    else: 
        # TODO: path can be url
        raise NotImplementedError(f"Uncaught loading instructions for type {file_location_type(glb_path)}")
    return glb_dict

def glb2trimesh(glb_dict:dict) -> Type[trimesh.base.Trimesh]:
    assert 'geometry'  in glb_dict
    vertices = []
    faces = []
    next_offset = 0 
    for segment in glb_dict['geometry']:
        shape = glb_dict['geometry'][segment]
        vertices.append(shape['vertices'])
        faces.append(shape['faces'] + next_offset)
        next_offset += len(shape['vertices'])
    vertices = np.vstack(vertices)
    faces = np.vstack(faces )
    # TODO: figure out how to load textures 
    mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)    
    return mesh
    
def glbfile2objfile(glb_path:str, out_path:str, 
                    transform:Callable = None)->str:
    
    # TODO: check that the GLB path exists. S3?
    # glb_dict = glbfile2glb(glb_path)
    mesh = glbfile2trimesh(glb_path) # glb2trimesh(glb_dict)

    if transform is not None:
        mesh = transform(mesh)

    if file_location_type(out_path) == 'local':
        prepare_dir_of(out_path)
        with open(out_path, 'w') as f:
            f.write(trimesh.exchange.obj.export_obj(mesh))
    else:
        raise NotImplementedError(f"Uncaught save instructions for type {file_location_type(out_path)}")

    return out_path

 