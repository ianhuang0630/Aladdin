"""
Implementations of different changes done to the scene shoppinglist, to study effects in downstream retrieval and editing
"""

from sem_complete.sem_complete import Brainstorm, SceneGraphs
import copy

def coarsify2abstract(shoppinglist: list[dict], abstract_scene_description: str) -> list[dict]:
    """ Replaces all class_name field with abstract scene description, and all attributes with empty string.
    """
    shoppinglist_ablated = copy.deepcopy(shoppinglist)
    for el in shoppinglist_ablated :
        assert 'class_name' in el and 'attributes' in el
        el['class_name'] = abstract_scene_description
        el['attributes'] = ''
    return shoppinglist_ablated 

def coarsify2object_class_name(shoppinglist: list[dict], abstract_scene_description: str) -> list[dict]:
    """ Replaces the attributes field with empty string.
    """
    shoppinglist_ablated = copy.deepcopy(shoppinglist)
    for el in shoppinglist_ablated :
        assert 'class_name' in el and 'attributes' in el
        el['attributes'] = ''
    return shoppinglist_ablated 

def coarsify2object_class_name_abstract(shoppinglist: list[dict], abstract_scene_description: str) -> list[dict]:
    """ Replaces the attributes field with the abstract scene description
    """
    shoppinglist_ablated = copy.deepcopy(shoppinglist)
    for el in shoppinglist_ablated :
        assert 'class_name' in el and 'attributes' in el
        el['attributes'] = abstract_scene_description
    return shoppinglist_ablated 



def densify2object_class_name_abstract_attributes(shoppinglist: list[dict], abstract_scene_description: str) -> list[dict]:
    """ inserts the abstract scene description into the object attributes
    """

    shoppinglist_densified = copy.deepcopy(shoppinglist)
    for el in shoppinglist_densified: 
        assert 'class_name' in el and 'attributes' in el
        el['attributes'] = f'in {abstract_scene_description}, ' + el['attributes']
    return shoppinglist_densified