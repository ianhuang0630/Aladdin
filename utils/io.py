"""
All things related to file storage
"""

import os
import requests
from bson.objectid import ObjectId
from typing import Type
from io import BytesIO
import json
import datetime
from PIL import Image, PngImagePlugin

def file_location_type(path:str) -> str:
    """ given a type, it returns a string indicating whether this is `local`, or `url`
    """
    if path.startswith('http://') or path.startswith('https://'):
        return 'url'
    return 'local' 

def prepare_dir(dir_location):
    if file_location_type(dir_location) == 'local' and len(dir_location):
        if not os.path.exists(dir_location):
            os.makedirs(dir_location)
    else:
        raise NotImplementedError(f'Uncaught file location type {fl_type}')
     

def prepare_dir_of(file_location):
    prepare_dir(os.path.dirname(file_location))
    return file_location


def get_event_file_location(session_token:str):
    return os.path.join(session_token, f'{datetime.datetime.now()}.json')


def read_img(file_location:str) -> Type[PngImagePlugin.PngImageFile]:
    """ Reads the  image back as PIL Image, given url or local path
    """
    fl_type = file_location_type(file_location) 
    if fl_type =='url':
        im = Image.open(BytesIO(requests.get(file_location).content))
    elif fl_type == 'local':
        if not os.path.exists(file_location): 
            raise FileNotFoundError(f'{file_location} does not exist.')
        im = Image.open(file_location)
    else:
        raise NotImplementedError(f'Uncaught file location type {fl_type}')
    return im

def read_event_file(file_location:str) -> dict:
    """ Event files loaded as json
    """
    with open(file_location, 'r') as f:
        values = json.load(f)
    return values

def write_event_file(file_location:str, values: dict): 
    """ Event files stored as json.
    """
    prepare_dir_of(file_location)
    if file_location_type(file_location) == 'local':
        with open(file_location, 'w') as f:
            json.dump(values,f)
    else:
        raise NotImplementedError(f'Uncaught file location type {fl_type}')
    return file_location

def temp_file(dir_path:str, extension:str) -> str:
    out_path = os.path.join(dir_path, str(ObjectId())+extension)
    prepare_dir_of(out_path)
    return out_path
