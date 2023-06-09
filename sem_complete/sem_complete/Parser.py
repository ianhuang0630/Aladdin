"""
Implementation of the parsing modules to turn language into scene information.
"""

def starListParser(input_list: str):
    """ Parser for string of list, with items that are separated by newlines, started with dashes.
    """ 
    # TODO tell when the list has ended
    input_list = input_list.strip().lower()
    items = [el.strip() for el in input_list.split('*')]
    items = [el for el in items if len(el) != 0]
    return items


def starKeyValParser(key2vals:str, keyval_separator:str=":", val_separator=",", val_parser=None):
    """ parser for string of list, with items that are separated by newlines, starting with dashes, 
    with keys and values separated by keyval_separator. 
    """
    # TODO tell when the list has ended
    key2vals = starListParser(key2vals)
    key2vals = [el.split(keyval_separator) for el in key2vals]

    if val_parser is None: 
        # if there is no additional parser that should be used to parse the values.
        items = {el[0].strip().lower() : [el.strip().lower() for el in ''.join(el[1:]).split(val_separator)] for el in key2vals}
    else:
        items = {el[0].strip().lower() : val_parser(''.join(el[1])) for el in key2vals}
    return items
    

def parseDescriptionCoordinates(val: str):
    """
    Parser for positional descriptions of this type:
    [description of position]  Coordinate(s): [coordinate 1] ... [coordinate N]
    with coordinate in floats. An example:

        on the couch. Coordinates (-1.0, 0.5) (-1.0, 1.5)
        in the corner of the room. Coordinate (-2.0, -1.0)

    """
    def coordinate_parse(coord_str):
        foo = ''.join(coord_str.split()).split('(')
        foo = [el for el in foo if len(el) > 0]
        foo = [el.strip().split(')')[0] for el in foo]
        foo = [tuple([float("".join(coord.split())) for coord in el.split(',')]) for el in foo]
        return foo
    val =  val.lower()
    if "coordinates" in val:
        value_splits = val.split("coordinates") 
        description = value_splits[0]
        instance_coordinates = coordinate_parse(value_splits[1])
        if len(value_splits) > 2: 
            raise ValueError(f"input, '{val}' does not follow the requested format!")
    elif "coordinate" in val: 
        value_splits = val.split("coordinate") 
        description = value_splits[0]
        instance_coordinates = coordinate_parse(value_splits[1])
        if len(value_splits) > 2: 
            raise ValueError(f"input, '{val}' does not follow the requested format!")
    else:
        raise ValueError(f"input, '{val}' does not follow the requested format!")
    return {'description': description, 'instance_coordinates': instance_coordinates}
