"""
New module for parsing dreams
"""
from typing import Type
from sem_complete.sem_complete.Brainstorm import GPTInterface, Brainstormer, IterativeBrainstormer
from sem_complete.sem_complete.SceneGraphs import SceneObjectClass, SceneShoppingList
from sem_complete.sem_complete.Template import SceneQueryTemplate
import time
from loguru import logger
from sem_complete.sem_complete.Parser import starListParser, starKeyValParser


class DreamEnvironment(object):
    def __init__(self, abstract_scene_description):
        self.description = abstract_scene_description 
        self.scene_shoppinglists = None

    def set_scene_shoppinglist(self, shoppinglist: Type[SceneShoppingList]):
        assert isinstance(shoppinglist, SceneShoppingList)
        self.scene_shoppinglists = shoppinglist
        return self


class DreamSegment(object):
    """ A single scene of a dream.
    """
    def __init__(self, description:str):
        self.description = description
        self.stray_objects = []
        self.environment = None

    def add_object(self, obj: Type[SceneObjectClass]):
        assert isinstance(obj, SceneObjectClass)
        self.stray_objects.append(obj)
        return self

    def set_environment(self, env: Type[DreamEnvironment]):
        assert isinstance(env, DreamEnvironment)
        self.environment = env
        return self
        
        
class DreamSegmentCollection(object):
    def __init__(self, description: str):
        self.description = description 
        self.scenes = []
    
    def add_scene(self, dream_segment:Type[DreamSegment]):
        self.scenes.append(dream_segment)



class Dreamstormer(Brainstormer):
    """ 
    """
    def __init__(self, interface=None, temperature=0.8, max_tokens=1024):
        super().__init__(gpt_interface=interface, 
                         temperature=temperature, 
                         max_tokens=max_tokens)
        self.iterative_brainstormer = IterativeBrainstormer(interface, temperature, max_tokens)
    
    def partition_scenes(self, description:str):
        """ 
        """

        sqt = SceneQueryTemplate('sem_complete/templates/dreams/forest_decompose.txt')
        question= f"Here's a description of someone's dream:\n\n{description}\n\n---\n\n"
        question+= "Here's the same dream but breaking up the description into paragraphs according to scenes:\n"
        prompt, conditioning, gpt_question = sqt.apply_directly(question)
        response = self.interface.query(conditioning, gpt_question)  
        scene_descriptions = [scene for scene in response.split('\n') if len(scene)]
        
        return scene_descriptions


    def scene_key_objects(self, fragment_description:str):
        sqt = SceneQueryTemplate('sem_complete/templates/dreams/forest_nodeattributes.txt')
        question = f"Here's a fragment of a dream:\n\n{fragment_description}\n\nHere is a list of the key inanimate physical objects found in the fragment, along with descriptions of their appearances:"
        prompt, conditioning, gpt_question = sqt.apply_directly(question)

        response = self.interface.query(conditioning, gpt_question) 
        categories = starKeyValParser(response)
        categories = {key: ', '.join(val) for key, val in categories.items()}
        return categories
         
    def abstract_scene_description(self, fragment_description:str):
        sqt = SceneQueryTemplate('sem_complete/templates/dreams/forest_abstractscenedescription.txt')
        question = f"Here's a fragment of a dream:\n\n{fragment_description}\n\nIn a simple phase, a description  of the environment/setting in the fragment would be:\n"
        prompt, conditioning, gpt_question = sqt.apply_directly(question)

        response = self.interface.query(conditioning, gpt_question) 
        return response.strip('.').strip().lower()

    def unique_scene_descriptions(self, description):
        sqt = SceneQueryTemplate('sem_complete/templates/dreams/bandroom_settings.txt')
        question = f"Here's a description of a person's dream:\n\n{description}\n\nHere's a list of the unique locations where this dream took place, each described using a phrase (visual appearance, style...etc):" 
        
        prompt, conditioning, gpt_question = sqt.apply_directly(question)
        response = self.interface.query(conditioning, gpt_question)
        abstract_scene_descriptions = starListParser(response) 
        return abstract_scene_descriptions


    def map2unique_environment(self, fragment_description, env_descriptions):
        
        sqt = SceneQueryTemplate('sem_complete/templates/dreams/forest_map2env.txt')
        question = f"Here's a fragment of a dream:\n\n{fragment_description}\n\nWhich of the following environments is this fragment taking place in?\n"
        for idx, envd in enumerate(env_descriptions):
            question += f'{idx+1}. {envd}\n'
        question += '\nAnswer: '
        
        prompt, conditioning, gpt_question = sqt.apply_directly(question)
        response = self.interface.query(conditioning, gpt_question) 
        if '.' in response:
            selected_index = int(response.split('.')[0])-1
        else:
            selected_index = int(response.strip('.').strip())-1
        return selected_index
            

    def run(self, description:str):
        # ask for places and things.
        logger.info(f"INPUT:\n{description}\n")

        # creates the per-environment description. Let's say there are K different
        # environments in the dream. 
        per_environment_descriptions = self.unique_scene_descriptions(description)    
        environments = [DreamEnvironment(env_desc) for env_desc in per_environment_descriptions]
        
        # each "Scene" is a fragment -- a paragraph within the dream.
        scene_descriptions = self.partition_scenes(description)
        
        # TODO: verify that the scene descriptions are concatenating to be the input description.
        dream_segments_collection = DreamSegmentCollection(description)
        
        for scene_idx, scene_desc in enumerate(scene_descriptions):
            logger.info(f"SCENE {scene_idx}: {scene_desc}") 
             
            # asd = self.abstract_scene_description(scene_desc)
            # logger.info(f"ABSTRACT SCENE DESCRIPTION: {asd}")
            envidx = self.map2unique_environment(scene_desc, per_environment_descriptions)
            
            
            dream_segment = DreamSegment(description=scene_desc)
            key_objects = self.scene_key_objects(scene_desc)

            # which environment are we in?
            for object_category in key_objects:
                logger.info(f"KEY ITEM: {object_category}: {key_objects[object_category]}")
                dream_segment.add_object(SceneObjectClass(class_name=object_category,
                                 instances=None,
                                 attributes=key_objects[object_category]))

            dream_segment.set_environment(environments[envidx]) 
            dream_segments_collection.add_scene(dream_segment)

        for env in environments: 
            logger.info(env.description)
            abstract_scene_shopping_list = self.iterative_brainstormer.run(env.description, num_iterations=1)
            env.set_scene_shoppinglist(abstract_scene_shopping_list)
        
        return dream_segments_collection
        
        
        