import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json

def get_Future3D_all_model_ids(cfg: DictConfig) -> dict:
    with open(cfg.data.future3d_json, 'r') as f:
        model_info = json.load(f)
    
    ids = [el['model_id'] for el in model_info]
    form = "future3d"
    return {'ids': ids, 'format': form}

# function that returns the path of a 3D model
def get_Future3D_model_path(cfg: DictConfig, id: str) -> dict:
    path = os.path.join(cfg.data.future3d, id, 'normalized_model.obj') # taking normalized
    form = "future3d-normalized" # .obj, .glb ..etc.
    return {'path': path, 'format': form}

def get_Future3D_visual_path(cfg: DictConfig, id: str) -> dict: 
    path = os.path.join(cfg.data.future3d, id, 'image.jpg') # taking normalized
    form = "future3d-jpg"
    return {'path': path, 'format': form}

def get_Future3D_text_annotation(cfg: DictConfig, id: str) -> dict:
    form = "future3d-STMCS"
    with open(cfg.data.future3d_json, 'r') as f:
        model_info = json.load(f)
    id2annot = {el['model_id']: [el['style'], el['theme'], el['material'], el['category'], el['super-category']] for el in model_info}
    text_annot = ' '.join([el for el in id2annot[id] if el is not None ]).lower() # we concatenate in the olrder of Style, Theme, Material, Category, SuperCategory
    return {'text': text_annot if len(text_annot) else None, 'format': form } 

if __name__=='__main__':
    @hydra.main(version_base=None, config_path="../../configs", config_name="data")
    def app(cfg: DictConfig):
        target_id = get_Future3D_all_model_ids(cfg=cfg)['ids'][0]
        text_annotations = get_Future3D_text_annotation(cfg=cfg, id=target_id)['text']
        print(text_annotations)        
        
        obj_path  = get_Future3D_model_path(cfg=cfg, id=target_id)['path']
        assert os.path.exists(obj_path)
        img_path = get_Future3D_visual_path(cfg=cfg, id=target_id)['path']
        assert os.path.exists(img_path)

    app()