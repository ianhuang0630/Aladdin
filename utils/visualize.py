"""
Visualize
"""

import matplotlib.pyplot as plt
from typing import Type, Callable
import numpy as np
import trimesh
from PIL import Image, PngImagePlugin
from utils.io import read_img
from utils.convert import objfile2trimesh, glbfile2trimesh

def print_shoppinglist(shoppinglist: list[dict]) -> str:
    out = ""
    for item_idx, el in enumerate(shoppinglist):
        out += (f"{item_idx} ### {el['instances']} {el['class_name']}, {el['attributes']}") + "\n"
    return out 
    

def viz_trimesh(obj_path:str, transform:Callable = None) -> Type[trimesh.Scene]:
    """
    Returns a trimesh scene.
    """
    
    if obj_path.endswith('.obj'):
        obj_mesh = objfile2trimesh(obj_path)
    elif obj_path.endswith('.glb'):
        obj_mesh = glbfile2trimesh(obj_path)
    else :
        raise NotImplementedError( f'Uncaught load for {obj_path}')

    if transform is not None:
        obj_mesh = transform(obj_mesh)
    # dl = trimesh.scene.lighting.DirectionalLight(name="directional_light", intensity=1000)
    scene = trimesh.Scene() #lights=[dl])
    scene.add_geometry(obj_mesh) 
    return scene

def viz_object_options(options: dict):
    
    num_options = len(options['results'])
    fig, axes = plt.subplots(nrows=1,ncols=num_options)
    for idx, (ax, option) in enumerate(zip(axes, options['results'])):
        viz_img(ax, img_path=option['image'], title=f"{idx}: {np.round(option['sim-score'], 2)}" )
    fig.suptitle(options['category'])
    return axes

def viz_single_img(img_path: str, **kwargs) -> Type[PngImagePlugin.PngImageFile]:
    fig, ax = plt.subplots()
    viz_img(ax, img_path, **kwargs)
    return ax

def viz_img(ax: Type[plt.axis], img_path: str, 
            title: str = None) -> Type[plt.axis]:
    try: 
        im = read_img(img_path)    
        ax.imshow(im)
    except:
        print(f'error loading/displaying image at {img_path}')
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax
