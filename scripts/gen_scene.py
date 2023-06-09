
import sys
import requests
from experimental.semantic_ablations import *
from utils.records import *
from loguru import logger
from tqdm import  tqdm


assert len(sys.argv) >= 2, 'Insufficent commandline arguments.'

session_token = sys.argv[1]
port = 5000
if len(sys.argv) >= 3: 
    abstract_scene_description = sys.argv[2]
if len(sys.argv) >= 4:
    port = sys.argv[3]

##########################
# check if the  session_token exists
# if exists, load previous state to know where to continue

# if not exist, starts new session.
if len(sys.argv) >= 5 and sys.argv[4] == 'baseline':
    # assumes that the previous session exists
    event_collection_prev = EventCollection().load_from_session('interaction_records/', session_token)

    session_token += '_baseline'

    shopping_list = event_collection_prev.filter_to('SHOPPING_LIST')[-1] 
    abstract_scene_description = event_collection_prev.filter_to('INPUT_SCENE_DESCRIPTION')[-1].scene_description
    num_object_categories = len(shopping_list.shoppinglist)
    
    coarse_shopping_list = coarsify2abstract(shopping_list.shoppinglist, 
                           abstract_scene_description=abstract_scene_description)
    coarse_shopping_list = coarse_shopping_list[0:1] # just on element
    print(num_object_categories)
    stage2_input = {'asset_property_list': coarse_shopping_list, 
                    'topk': num_object_categories, 
                    'session_token': session_token}

    ##########################
    stage2_output = requests.post(f'http://localhost:{port}/retrieve/', json=stage2_input)
    stage2_output_json = stage2_output.json()

    for texturify_option in tqdm(range(len(stage2_output_json[0]['results']))):
        texturify_obj = 0 
        stage3_input = {'objects': stage2_output_json,
                'object_index': texturify_obj,
                'option_index': texturify_option,
                'session_token': session_token,
                'preview': True}
        try:
            stage3_output = requests.post(f'http://localhost:{port}/stylize/', json=stage3_input)
            logger.info(stage2_output_json[texturify_obj]['query']) 
            logger.info(stage3_output.json())
        except:
            logger.info(f"ERROR with texturing object [{stage2_output_json[texturify_obj]['query']}]")
            continue


else:

    stage1_input = {'abstract_scene_description': abstract_scene_description, 
                    'recursion_levels': 1, 'session_token': session_token}
    stage1_output = requests.post(f'http://localhost:{port}/semcomplete/', json=stage1_input)
    densify_stage1_output = densify2object_class_name_abstract_attributes(stage1_output.json(), 
                                                                        abstract_scene_description)
    stage2_input = {'asset_property_list': densify_stage1_output, 'topk': 1, 'session_token': session_token}


    stage2_output = requests.post(f'http://localhost:{port}/retrieve/', json=stage2_input)
    stage2_output_json = stage2_output.json()



    for texturify_obj in tqdm(range(len(stage2_output_json))):
        texturify_option = 0 # selecting the first one by default
        stage3_input = {'objects': stage2_output_json,
                'object_index': texturify_obj,
                'option_index': texturify_option,
                'session_token': session_token,
                'preview': True}

        try:
            stage3_output = requests.post(f'http://localhost:{port}/stylize/', json=stage3_input)
            logger.info(stage2_output_json[texturify_obj]['query']) 
            logger.info(stage3_output.json())
        except:
            logger.info(f"ERROR with texturing object [{stage2_output_json[texturify_obj]['query']}]")
            continue

# perhaps load from previous sesison

