import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
from loguru import logger
import hydra
from omegaconf import DictConfig
from bson.objectid import ObjectId
import shutil
import openai

app = Flask(__name__)
#CORS(app)
import os
import sys
sys.path.append('../')
from sem_complete.sem_complete import Brainstorm, SceneGraphs

# shape retrieval
from shape_retrieve.DatasetInterface import DatasetInterface, DatasetEnsembleInterface
from shape_retrieve.data.future3d import get_Future3D_all_model_ids, get_Future3D_model_path, get_Future3D_visual_path, get_Future3D_text_annotation
from shape_retrieve.data.allenai_objaverse import get_objaverse_all_model_ids, get_objaverse_model_path, get_objaverse_visual_path, get_objaverse_text_annotation
from shape_retrieve.feature_extractor.base import GenericEncoder, VisuoLinguisticEncoder
from shape_retrieve.feature_extractor.vit import VIT_encoder
from shape_retrieve.retriever.base import GenericRetriever
from shape_retrieve.retriever.cossine_sim import CossineSim

# Texturing
from texturize.TexturingInterface import TexturingInterface

from utils.io import * 
import utils.aws_buckets as buckets
from utils.convert import *
from utils.mesh_transform import *
from utils.records import * 


from hydra import compose, initialize
initialize(version_base=None, config_path="../configs", job_name="test_app")
cfg = compose(config_name="data")


event_file = lambda x: get_event_file_location(os.path.join(cfg.general.record_dir, x))

dataset_interface = None
retriever = None
feature_extactor = None

def dataset_setup():
    logger.info('Setting up DatasetInterface...')

    global retriever
    global feature_extractor 
    feature_extractor = VIT_encoder(device=cfg.retrieval.device)
    retriever = CossineSim(device=cfg.retrieval.device)

    global dataset_interface
    # setup for the data...
    dataset_output_jsons = []
    dataset_setups = []
    dataset_setups.append(
        {'model_path': get_Future3D_model_path,
        'visual_path': get_Future3D_visual_path,
        'text_annotation': get_Future3D_text_annotation,
        'all_model_ids': get_Future3D_all_model_ids}
    )
    dataset_output_jsons.append(
        'shape_retrieve/datasets/future3d_img_vit.json',
    )

    dataset_setups.append(
        {'model_path': get_objaverse_model_path,
        'visual_path': get_objaverse_visual_path,
        'text_annotation': get_objaverse_text_annotation, 
        'all_model_ids': get_objaverse_all_model_ids} #getting
    ) 
    dataset_output_jsons.append(
        'shape_retrieve/datasets/objaverse30000_img_vit.json' 
    )

    logger.info("setting up DatasetEnsemble Interface...")

    dataset_interface = DatasetEnsembleInterface(cfg, 
                    [DatasetInterface(cfg,
                    get_model_path=setup['model_path'],
                    get_visual_path=setup['visual_path'],
                    get_text_annotation=setup['text_annotation'],
                    get_all_model_ids=setup['all_model_ids']).extract_features(
                        feature_extractor=feature_extractor,
                        output_json=output_json ,
                        override_json=False)
                    for setup, output_json  in zip (dataset_setups, dataset_output_jsons)
                    ])

    logger.info('Done!')

dataset_setup()


# Huggingface login
from huggingface_hub import login
with open('credentials/huggingface_key', 'r') as f:
    huggingface_token = f.readline().strip()
    login(token=huggingface_token)

# Create the GPTInterface 
with open('credentials/openai_key', 'r')  as f:
    token = f.readlines()
    token = token[0].strip()
logger.info(f"Attempting to initialize GPT interface.")
myGPTinterface = Brainstorm.GPTInterface(api_key=token,
                                default_temperature=0.8,
                                default_max_tokens=1024)
logger.info(f"Successfully intiated GPT interface.")


@app.route("/", methods=['GET'])
def hello_world() -> None :
    print(cfg) 
    return "<p>Hello, World!</p>"


@app.route("/semcomplete/", methods=["POST"])
def semantic_autocomplete():
    """
    Inputs:
        abstract_scene_description: abstract scene description as a string
        recursion_level : the number of recursions done for the brainstormer 
    Returns: a list of dictionaries, with category name, # instances and attributes.
    """
    # parsing inputs
    inputs = json.loads(request.data)
    scene_description = inputs['abstract_scene_description']
    
    recursion_levels = None
    if 'recursion_levels' in inputs:
        recursion_levels = inputs['recursion_levels']
    
    session_token = inputs['session_token']
    # intialize input and output 
    input_event = InputSceneDescription(session_token, scene_description)
    input_event_file = input_event.save_to_event_file(event_file(session_token))

    output_event =  ShoppingList().tick()

    # semantic autocomplete
    try:
        brainstormer = Brainstorm.IterativeBrainstormer(interface=myGPTinterface)
        output = brainstormer.run(scene_description, 1) 
    except openai.error.RateLimitError:
        logger.info('language model overloaded, try again later')
        return 'language model overloaded, try again later', 500
    except:
        logger.info('language model produced unparsable output')
        return 'language model produced unparsable output', 500
        
    assert isinstance(output, SceneGraphs.SceneShoppingList)
    # flattening hierarchy
    output = output.flatten()
    output = [el.to_json(recursive=False)  for el in output]

    # finalize output
    output_event.tock().update(output)
    output_event_file = output_event.save_to_event_file(event_file(session_token))
    
    return jsonify(output)


@app.route("/retrieve/", methods=["POST"])
def retrieve():
    """
    Inputs: 
        asset_property_list: the list output from semcomplete ( a list of dictionaries )     
    Returns:
        a list of options (list of dictionaries)
    """

    inputs = json.loads(request.data) 
    
    asset_property_list = inputs['asset_property_list'] # same type of output of semantic_autocomplete.
    topk = inputs['topk']

    session_token = inputs['session_token']
    input_event = ShoppingListEdit(session_token, shoppinglist=asset_property_list)
    input_event_file = input_event.save_to_event_file(event_file(session_token))
                
    output_event = RetrievedOptions(session_token).tick()

    if dataset_interface is None:
        dataset_setup()

    objs = [SceneGraphs.SceneObjectClass().from_json(asset, recursive=False) for asset in asset_property_list]
    queries = [f'{obj.class_name}, {obj.attributes}' for obj in objs]
    
    results = dataset_interface.language_retrieve(queries, topk=topk, 
                    retriever=retriever, feature_extractor=feature_extractor)
    final_result = []
    for d, o, r in zip (queries, objs, results):
        res = {'query': d,
               'description': o.attributes,
               'category': o.class_name,
               'instances': o.instances,
               'topk': topk,
               'results': r}
        img_paths = dataset_interface.img_path(r) 
        model_paths = dataset_interface.model_path(r)
        # TODO: may need to temporarily copy for file to be accessible..

        if cfg.server.backup_img:
            new_paths = []
            for img_path in img_paths:
                src = img_path['path']
                dst = os.path.join(cfg.server.img_download_root, 
                                   str(ObjectId()))
                os.makedirs(dst, exist_ok=True)
                dst = os.path.join(dst, os.path.basename(src))
                shutil.copy(src, dst)
                new_paths.append(dst)
            img_paths = new_paths
        else:
            img_paths = [el['path'] for el in img_paths]

        if cfg.server.backup_mesh:
            new_paths = []
            for model_path in model_paths:
                src = model_path['path']
                dst = os.path.join(cfg.server.model_download_root, 
                                   str(ObjectId()))
                os.makedirs(dst, exist_ok=True)
                dst = os.path.join(dst, os.path.basename(src))
                shutil.copy(src, dst)
                new_paths.append(dst)
            model_paths = new_paths
        else:
            model_paths = [el['path'] for el in model_paths]

        for rr, imgp, modp in zip(r, img_paths, model_paths):
            rr['image'] = imgp 
            rr['model'] = modp

        final_result.append(res)

    # finalize_output 
    output_event.tock().update(final_result)
    output_event_file = output_event.save_to_event_file(event_file(session_token))
    return jsonify(final_result)

@app.route("/stylize/", methods=["POST"])
# @hydra.main(version_base=None, config_path='../configs', config_name="data")
def stylize():
    """
    Inputs:
        objects: output of `retrieve`.
        object_index: decides the object category to texturize.
        option_index: decides the option within the aforementioned object category to texturize.
        preview  (optional): True when just want the first view. 
    Returns:
        download links for (1) result folder, (2) obj, (3) mtl, (4) albedo png.
    """
    inputs = json.loads(request.data)
    session_token = inputs['session_token']
    objects = inputs['objects']  # outputs of retrieve
    object_index = inputs['object_index'] # which object category
    option_index = inputs['option_index'] # which choice

    preview = None
    if 'preview' in inputs:
        preview = inputs['preview'] # boolean :  True when just want the first image.
    backdrop = None if 'backdrop' not in inputs else inputs['backdrop']
    guidance_scale = None if 'guidance_scale' not in inputs else inputs['guidance_scale']

    # NOTE: if we have multiple GPU's, we should parallelize it! 
    # that means allocating a GPU that isn't super loaded.
    if object_index >= len(objects):
        return f"object index {object_index} out of range for object list of length {len(objects)}", 500

    if option_index >= len(objects[object_index]['results']):
        return f"option index {option_index} out of range for option list of length {len(objects[object_index])}", 500

    # TODO : provide override option.
    query = objects[object_index]['query']
    shape_path = objects[object_index]['results'][option_index]['model']
    shape_info = objects[object_index]['results'][option_index]

    input_event = RetrievalSelection(session_token, selection={
        'query': query, 'shape_path': shape_path, 'shape_info': shape_info,
        'object_index': object_index, 'option_index': option_index})
    input_event_file = input_event.save_to_event_file(event_file(session_token))

    output_event = TexturedObject(session_token).tick()

    # convert if needed
    if shape_path.endswith('.glb'):
        logger.info(f'Converting {shape_path}...') 
        # transformations involve a rescaling, recentering and rotation.
        shape_path = glbfile2objfile(shape_path, 
                                    temp_file(cfg.general.cache_dir, extension='.obj'),
                                    transform=lambda mesh: global_rotation(recenter(resize_to_unit_cube(mesh),  # recenter and rescale into unit cube
                                                                        center_fn = get_center_bbox),
                                                                        rotation_matrix=np.array([[1,0,0],
                                                                                                  [0,0,1],
                                                                                                  [0,-1,0]]) 
                                                                        ))
        logger.info(f'Done! Saved to {shape_path}.')

    object_id = str(ObjectId())
    query_cmd_string = '\"'+query+'\"' 
    cmd = ["python", "texturize/TexturingInterface.py", 
                    f"+texturing.query_instance={ query_cmd_string}",
                    f"+texturing.shape_path_instance={str(shape_path)}",
                    f"+texturing.backdrop_instance={str(backdrop)}",
                    f"+texturing.guidance_scale_instance={str(guidance_scale)}",
                    f"+texturing.object_id_instance={str(object_id)}"]
    
    try: 
        subprocess.run(cmd, timeout = cfg.texturing.timeout) 
    except subprocess.TimeoutExpired:
        logger.info('Texturing of {} expired! Moving on.')
        return "Subprocess call timed out.", 500

    # THe BELOW is put into  
    # ################################## 
    # interface = TexturingInterface(cfg)
    # interface.attach_text(query) # language instruction for appearance
    # # setup for this query...
    # interface.attach_shape_path(shape_path) # tell it which shape_path to attend to
    # interface.attach_backdrop_path(backdrop)
    # interface.set_guidance_scale(guidance_scale)
    # interface.set_exp_root(cfg.server.mesh_download_root)
    # interface.set_name(object_id)
    # interface.set_seed(cfg.general.seed) 
    # interface.run() 
    # ##################################    

    # intended mesh paths
    directory_path = os.path.join(cfg.server.mesh_download_root, object_id)
    mesh_obj = os.path.join(directory_path, 'mesh', 'mesh.obj')
    mesh_mtl = os.path.join(directory_path, 'mesh', 'mesh.mtl')
    mesh_alb = os.path.join(directory_path, 'mesh', 'albedo.png')
    
    mesh_output =  {'top_directory': directory_path,
                    'obj': mesh_obj,
                    'mtl': mesh_mtl,
                    'albedo': mesh_alb}

    # TODO: upload to aws, if ordered by config!
    if cfg.server.bucket_mesh:
        logger.info('Beginning upload procedure to S3 buckets!')
        # upload and override.
        output_id = str(ObjectId())
        
        shutil.make_archive(os.path.join(cfg.general.cache_dir, output_id), 
                            'zip', os.path.join(directory_path, 'mesh')) 
        logger.info(f'Compressed {directory_path}/mesh.')
        download_url = buckets.upload_file (cfg.server.mesh_download_aws_bucket,
                             os.path.join(cfg.general.cache_dir, f'{output_id}.zip'),
                             f'{output_id}.zip')
        logger.info(f'Uploaded to {download_url}')
        mesh_output['aws_bucket_zip'] = download_url
        logger.info('Finished upload procedure to S3 buckets')

    output_event.tock().update(mesh_output)
    output_event_file = output_event.save_to_event_file(event_file(session_token))

    return jsonify(mesh_output)


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
        app.run(host="localhost", port=port)
    else:
        app.run()
    # app.run(debug=True)
