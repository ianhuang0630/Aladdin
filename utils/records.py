"""
Module for recording I/O
"""
from __future__ import annotations
from loguru import logger
import json
from datetime import datetime
import time
from utils.io import *
from bson import ObjectId
import numpy as np
from sem_complete.sem_complete import SceneGraphs
import shutil
from typing import Union


class EventCollection(object):
    """ Class used for data processing from event files
    """
    def __init__(self):
        self.id = str(ObjectId())
        self.events = []

    def load_from_session(self, event_dir:str, session_token:str) -> EventCollection:
        if file_location_type (event_dir) == 'local':
            assert os.path.exists(os.path.join(event_dir, session_token)), f'{os.path.join(event_dir, session_token)} does not exist'
            
            for event_file in os.listdir(os.path.join(event_dir, session_token)):
                # try:
                new_event = Event().load_from_event_file(os.path.join(event_dir, session_token, event_file))
                # except:
                #     logger.info(f'Error loading from {os.path.join(event_dir, session_token, event_file)}')
                #     continue
                
                self.add_event(new_event)
                    
        else:
            raise NotImplementedError(f'Uncaught file location {file_location_type(event_dir)}')

        self.time_sorted()
        return self

    def add_event(self, event:Type[Event]) -> EventCollection:
        assert isinstance(event, Event), f"invalid type {type(event)} for EventCollection"
        self.events.append(event)
        return self

    def time_sorted(self) -> EventCollection:
        self.events = sorted(self.events)
        return self

    def __str__(self) -> str:
        printout = ""
        for ev in sorted(self.events):
            printout += str(ev)+'\n'
        return printout

    def zip_all_textured_objects(self, output_zip:str, cache_dir='cache/'):

        # TODO: check locality of output_zip
        if not output_zip.endswith('.zip'):
            random_output_zip_id = str(ObjectId())
            output_zip = os.path.join(output_zip, f'{random_output_zip_id}')

        collection_id = os.path.basename(output_zip)# [:-len('.zip')]
        cache_dir = os.path.join(cache_dir, collection_id)
        prepare_dir_of(output_zip)
        prepare_dir(cache_dir)

        textured_outputs = sorted(self.filter_to('TEXTURED_OBJECT'))

        for ev in textured_outputs:
            # check if the output is actually there
            assert ev.type == 'TEXTURED_OBJECT' 
            
            # TODO: change this to be robust to urls
            obj_path = ev.textured_output_paths['obj']
            if not os.path.exists(obj_path):
                logger.info(f'{obj_path}: Not found! Moving on to other shapes')
                continue

            # if not, give error message and quit            
            src_mesh = os.path.dirname(ev.textured_output_paths['obj'])
            model_id = src_mesh.split('/')[-2]
            
            # get model id
            shutil.copytree(src_mesh, os.path.join(cache_dir, model_id ))
            logger.info(f'model {model_id} copied to {os.path.join(cache_dir, model_id)}')
            
        shutil.make_archive(output_zip, 'zip', cache_dir) 
        logger.info(f'zip prepared at {output_zip}.zip!')

        return output_zip
        
             

    def filter_to(self, label:Union[str, list[str]]):
        
        if isinstance(label, str):
            label = [label]

        for l in label: 
            assert l in ('EVENT', 'USER_INTERACTION', 'SYSTEM_RESPONSE',
                         'INPUT_SCENE_DESCRIPTION', 'SHOPPING_LIST',
                         'SHOPPING_LiST_EDIT', 'RETRIEVED_OPTIONS',
                         'RETRIEVAL_SELECTION', 'TEXTURED_OBJECT', 
                         'CHECKOUT'), f"unknown type {label}"

            if label in ('EVENT', 'USER_INTERACTION', 'SYSTEM_RESPONSE') :
                raise NotImplementedError(f"Currently not handling {label}")

       
        filtered_events = [ev for ev in self.events if ev.type in label]
        return filtered_events

    def get_texturing_timeline(self) -> list[Type[Transaction]]:
        events = sorted(self.filter_to(['RETRIEVAL_SELECTION', 'TEXTURED_OBJECT']))
        texturing_timeline = []
        for idx, ev in enumerate(events):
            if ev.type == 'TEXTURED_OBJECT': 
                assert idx >= 1, 'found a textured object without input event file??'
                input_event = events[idx - 1]
                output_event = ev
                texturing_timeline.append(Transaction(input_event, output_event))
        return texturing_timeline
    
    def get_state(self) -> dict:
        if len(self.events) == 0:
            raise ValueError("Have you loaded from session? Empty session data.")
        
        states =  {}
        for ev in sorted(self.events):
            ev_state = ev.export()
            if ev_state['type'] == 'SHOPPING_LIST_EDIT':
                # renaming
                ev_state['shoppinglist_edit']  = ev_state['shoppinglist']
                del ev_state['shoppinglist']
            states.update(ev_state)
         
        # latency, timestamp and _start_time, _end_time are irrelevant
        if 'latency' in states: del states['latency']
        if '_start_time' in states: del states['_start_time']
        if '_end_time' in states: del states['_end_time']


        out = {}
        meta = {} 
        
        meta['latest_update'] = states['timestamp']      
        meta['latest_event_type'] = states['type']
        meta['session_token'] = states['session_token'] 
        out['metadata'] = meta
        
        if 'scene_description' in states:
            out['stage1_input'] = states['scene_description']

        if 'shoppinglist'  in states:
            out['stage1_output'] = states['shoppinglist']

        if 'shoppinglist_edit' in states:
            out['stage2_input'] = states['shoppinglist_edit']

        if 'retrieved_results' in states:
            out['stage2_output'] = states['retrieved_results']
        
        if 'query' in states and 'shape_path' in states and 'shape_info' in states and  'object_index' in states and 'option_index' in states:
            out['stage3_input'] = {'query':  states['query'], 'shape_path': states['shape_path'], 'shape_info': states['shape_info'],
                            'object_index': states['object_index'], 'option_index': states['option_index']}

        if 'textured_output_paths' in states:
            out['stage3_output'] = states['textured_output_paths']

        return out

class Transaction(object):
    def __init__(self, input_event: Type[Event], output_event: Type[Event]):
        self.input_event = input_event
        self.output_event = output_event

    def __str__(self):
        return  f"INPUT: {self.input_event}\n" + f"OUTPUT: {self.output_event}"
        

class Event(object):
    def __init__(self, session_token:str = None):
        self.session_token = session_token
        self.timestamp = str(datetime.datetime.now())
        self.type = 'EVENT'

    # comparison methods for sorting
    def __lt__(self, obj: Event) -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp < obj.timestamp
    def __gt__(self, obj: Event) -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp > obj.timestamp
    def __le__(self, obj: Event) -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp <= obj.timestamp
    def __ge__(self, obj: Event) -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp >= obj.timestamp
    def __eq__(self, obj: Event) -> bool:
        if not isinstance(obj, Event): raise ValueError(f'Invalid comparison of type {type(obj)}')
        return self.timestamp == obj.timestamp 

    def load_from_event_params(self, **kwargs) -> Event: 
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return self

    def load_from_event_file(self, filepath: str) -> Event:
        fields = read_event_file(filepath) 
        return TYPE2CLASS[fields['type']]().load_from_event_params(**fields) 

    def export(self) -> dict:
        return vars(self)

    def save_to_event_file(self, filepath: str) -> str:
        attr_dict = self.export() 
        if hasattr(self, 'latency') and self.latency is None:
            raise ValueError("Forgot to call .tick and .tock?")
        write_event_file(filepath, attr_dict)
        return filepath

    @property
    def description(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return f'{str(self.timestamp)}: [{self.type}] -- {self.description}'


class UserInteraction(Event):
    """ Base class for all user interactions
    """
    def __init__(self, session_token: str = None):
        super().__init__(session_token)
        self.type = 'USER_INTERACTION'
     
    @property
    def description(self) -> str:
        return None


class SystemResponse(Event):
    """ Base class for all system responses
    """
    def __init__(self, session_token: str = None):
        super().__init__(session_token)
        self.type = 'SYSTEM_RESPONSE'
        self.latency = None

    def tick(self):
        self._start_time = time.time()
        return self

    def tock(self):
        assert hasattr(self, "_start_time"), "Did you forget to run .tick() ?"
        self._end_time = time.time()
        self.latency = self._end_time - self._start_time
        return self
        
    @property
    def description(self) -> str:
        return None

    def __str__(self) -> str:
        return f'{str(self.timestamp)} [latency {np.round(self.latency, 2)} s]: [{self.type}] -- {self.description}'
        
###### Specific implementations of different events below

class InputSceneDescription(UserInteraction):
    """ Includes information for original scene input
    """
    def __init__(self, session_token:str = None, abstract_scene_description:str = None):
        super().__init__(session_token) 
        self.scene_description = abstract_scene_description
        self.type = 'INPUT_SCENE_DESCRIPTION'
    
    @property
    def description(self) -> str:
        return f'[[ {self.scene_description} ]]'


class ShoppingList(SystemResponse):
    def __init__(self, session_token:str = None, shoppinglist: list[dict] = None):
        super().__init__(session_token)
        self.shoppinglist = shoppinglist
        self.type = 'SHOPPING_LIST'

    def update(self, shoppinglist: list[dict]) -> ShoppingList:
        self.shoppinglist = shoppinglist
        return self

    def stringify(self) -> str:
        stringed = ""
        for element in self.shoppinglist:
            class_name = element['class_name']
            attributes = element['attributes']
            stringed += f"{class_name}: {attributes}\n" 
        return stringed

    @property
    def description(self) -> str:
        return f'[[ List of {len(self.shoppinglist)} items ]]'


class ShoppingListEdit(UserInteraction):
    def __init__(self, session_token:str = None, shoppinglist: list[dict] = None):
        super().__init__(session_token)
        self.shoppinglist = shoppinglist
        self.type = 'SHOPPING_LIST_EDIT'
   
    @property
    def description(self) -> str:
        return f'[[ Edited list of {len(self.shoppinglist)} ]]'


class RetrievedOptions(SystemResponse):
    def __init__(self, session_token:str = None, retrieved_results:list = None):
        super().__init__(session_token)
        self.retrieved_results = retrieved_results
        self.type = 'RETRIEVED_OPTIONS'

    def update(self, retrieved_results:list = None):
        self.retrieved_results = retrieved_results
        
    @property
    def description(self) -> str:
        return f'[[  ]]'



class RetrievalSelection(UserInteraction):
    """ Happens right before texturing
    """
    def __init__(self, session_token:str = None, selection: dict = None):
        super().__init__(session_token)
        if selection is not None:
            for key in selection:
                setattr(self, key, selection[key])
        self.type = 'RETRIEVAL_SELECTION'

    @property
    def description(self) -> str:
        return self.query

class TexturedObject(SystemResponse):
    """ Happens right after Retrieval selection
    """
    def __init__(self, session_token:str = None, textured_output: dict = None):
        super().__init__(session_token) 
        self.textured_output_paths = textured_output
        self.type = 'TEXTURED_OBJECT'    

    def update(self, textured_output: dict = None):
        self.textured_output_paths = textured_output
        return self

    def render(self):
        raise NotImplementedError

    @property
    def description(self) -> str:
        return f"[[ mesh generated at {self.textured_output_paths['obj']} ]]"



class Checkout(UserInteraction):
    def __init__(self, session_token:str = None, saved_options: list = None):
        super().__init__(session_token)
        self.saved_options = saved_options 
        self.type = 'CHECKOUT'

    @property
    def description(self) -> str:
        return None


TYPE2CLASS = {
'EVENT': Event, 'USER_INTERACTION': UserInteraction, 'SYSTEM_RESPONSE':SystemResponse,
'INPUT_SCENE_DESCRIPTION': InputSceneDescription, 'SHOPPING_LIST': ShoppingList,
'SHOPPING_LIST_EDIT':  ShoppingListEdit, 'RETRIEVED_OPTIONS': RetrievedOptions,
'RETRIEVAL_SELECTION': RetrievalSelection, 'TEXTURED_OBJECT': TexturedObject, 
'CHECKOUT': Checkout}
