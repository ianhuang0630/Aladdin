"""
Implementation of data structures used to represent scenes, objects, and their attributes.
"""
import unittest
from typing import Type

class SceneObjectClass(object):
    """ ObjectClasses found within the scenes, NOT instances.
    """
    def __init__(self, class_name:str=None, instances:int=None, attributes:str = None):
        """
        Args:
            class_name: name of the object class
            instances: number of instances
            attributes: a string containing verbal description of its properties.
        """
        self.class_name = class_name 
        self.instances = instances
        self.attributes = attributes

        self.proximity_children = []

    def set_instances(self, instances:int):
        self.instances = instances
    def set_attributes(self, attributes:str):
        self.attributes = attributes

    def add_proximity_children(self, objectclass:'SceneObjectClass'):
        self.proximity_children.append(objectclass)
    
    def in_subtree(self, objectclass:'SceneObjectClass'):
        return objectclass == self or any([el.in_subtree(objectclass) for el in self.proximity_children]) 

    def print_subtree(self):
        # print out like this
        # | [node 1] 
        # | | [node 1 child 1 ]
        # | | [node 1 child 2]
        # | [node 2]
        # | | [node 2 child 1]
        # ... 
        children_printouts = []
        children_printouts.append(str(self))
        for el in self.proximity_children: 
            str_form, list_form =  el.print_subtree()
            for ell in list_form:
                children_printouts.append('| ' + ell)
            
        return '\n'.join(children_printouts), children_printouts

    def __str__(self):
        return "{} x {} : {}".format(self.instances, self.class_name, self.attributes)


    def to_json(self, recursive:bool=True):
        out = {'class_name': self.class_name,
            'instances' : self.instances,
            'attributes': self.attributes}
        if recursive: 
            out['proximity_children'] = [el.to_json() for el in self.proximity_children]
        return out

    def from_json(self, dictionary:dict, recursive:bool=True):
        self.class_name = dictionary['class_name']
        self.instances = dictionary['instances']
        self.attributes =  dictionary['attributes']
        if recursive:
            self.proximity_children = [SceneObjectClass().from_json(el) for el in dictionary['proximity_children']]
        return self

    def __eq__(self, other):
        """ Two of these object classes are equal to eachother if the subtree they span are the same
        """
        return (isinstance(other, SceneObjectClass)
        and self.class_name == other.class_name        
        and self.instances == other.instances
        and self.attributes == other.attributes
        and len(self.proximity_children) == len(other.proximity_children)
        and all([el1 == el2 for el1, el2 in zip(self.proximity_children, other.proximity_children)])
        )

    def bfs_enumeration(self):
        """ Breadth first enumeration of the elements of this subtree
        """        
        enumeration = [self]
        for el in self.proximity_children:
            enumeration.extend(el.bfs_enumeration())
        return enumeration

class SceneShoppingList(object):
    """ An assortment of different object classes, 
    with proximity relations to indicate parent-children structure.
    """
    def __init__(self,):
        self.root = SceneObjectClass("root", 1, "")
    
    def add_objectclass(self, 
                        objectclass:SceneObjectClass, 
                        parent:SceneObjectClass):
        # add to top-level if it isn't already in the tree
        if not self.root.in_subtree(parent):
            self.root.add_proximity_children(objectclass)
        else:
            parent.add_proximity_children(objectclass)

    def in_list(self, objectclass):
        return self.root.in_subtree(objectclass)

    def __str__ (self):
        printout, _ = self.root.print_subtree()
        return printout

    def to_json(self):
        return  [el.to_json() for el in self.root.proximity_children]

    def __eq__(self, other):
        return isinstance(other, SceneShoppingList) and self.root == other.root

    def flatten(self):
        """
        A breadth-first-search of the object tree.
        """
        enumeration = self.root.bfs_enumeration()
        enumeration = enumeration[1:] # removing the root
        return enumeration

if __name__=='__main__':
    unittest.main()
        




