import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
import objaverse 
from subprocess import DEVNULL, STDOUT, check_call

def get_objaverse_all_model_ids(cfg: DictConfig) -> dict:
    ids = objaverse.load_uids()
    form = 'objaverse'
    return {'ids': ids, 'format': form}

def get_objaverse_model_path(cfg: DictConfig, id: str) -> dict:
    # TODO: give a batch form of this

    ann=objaverse.load_annotations([id])[id]

    #  download if it doesn't exist
    with open(cfg.data.objaverse_json, 'r') as f:
        model2path = json.load(f)
    
    if id not in model2path or not os.path.exists(os.path.join(cfg.data.objaverse, 
                model2path[id]) ):
        if cfg.data.objaverse_download_missing:
            # let's see if we can download it and return the path
            objects = objaverse.load_objects(
                uids=[id],
                download_processes=1 # processes,
            )         
            path = objects[id]
        else:
            # report that the  file doesn't exist
            raise FileNotFoundError(f'objaverse {id} was not downloaded') 
    else:
        path = os.path.join( cfg.data.objaverse, model2path[id] )
    form = "objaverse-glb"
    return {'path': path, 'format': form}


def get_objaverse_visual_path(cfg: DictConfig, id: str) -> dict:
    # TODO: give a batch form of this

    if not os.path.exists(cfg.data.objaverse_thumbnail_download_path):
        os.makedirs(cfg.data.objaverse_thumbnail_download_path)

    ann = objaverse.load_annotations([id])[id]
    img_url = ann['thumbnails']['images'][0]['url']
    extension = img_url.split('.')[-1]

    if cfg.data.objaverse_lazy_thumbnail_download:
        form = f"objaverse-url"
        path = img_url # return the img_url
    else:
        form = f"objaverse-{extension}"
        path = os.path.join(cfg.data.objaverse_thumbnail_download_path, id+'.'+extension)
        if not os.path.exists(path):
            try :
                check_call(['wget', img_url, '-O', path], stdout=DEVNULL, stderr=STDOUT)    
            except:
                raise ValueError(f"Error getting {img_url}")
    return {'path': path, 'format': form}    
    
def get_objaverse_text_annotation(cfg: DictConfig, id: str) -> dict:
    ann = objaverse.load_annotations([id])[id]
    form = "objaverse-concat-categories" 
    # if len(ann['categories']):
    #     import ipdb; ipdb.set_trace()
    text_annot = ' '.join([el['name'] for el in ann['categories']]).lower() # for the objaverse dataset, we just concatenate the category information.
    return {'text': text_annot if len(text_annot) else None, 'format': form } 

if __name__=='__main__':
    @hydra.main(version_base=None, config_path="../../configs", config_name="data")
    def app(cfg: DictConfig):
        target_id = get_objaverse_all_model_ids(cfg=cfg)['ids'][0]
        text_annotations = get_objaverse_text_annotation(cfg=cfg, id=target_id)['text']
        path = get_objaverse_model_path(cfg=cfg, id=target_id)['path']
        path = get_objaverse_visual_path(cfg=cfg, id=target_id)['path']

    app()
    

