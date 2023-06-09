"""
This script iterates through the FUTURE3D dataset, extracts and stores features. 
"""
from shape_retrieve.data.future3d import get_Future3D_all_model_ids, get_Future3D_model_path, get_Future3D_visual_path, get_Future3D_text_annotation
from shape_retrieve.DatasetInterface import DatasetInterface
from shape_retrieve.feature_extractor.vit import VIT_encoder
import hydra

if __name__ == '__main__':
    
    @hydra.main(version_base=None, config_path="../../configs", config_name="data")
    def app(cfg: DictConfig):
        feature_extractor = VIT_encoder()        
        dataset_interface = DatasetInterface(cfg, 
                    get_model_path=get_Future3D_model_path,
                    get_visual_path=get_Future3D_visual_path,
                    get_text_annotation=get_Future3D_text_annotation,
                    get_all_model_ids=get_Future3D_all_model_ids)
        dataset_interface.extract_features(feature_extractor=feature_extractor,
                                           output_json='shape_retrieve/datasets/future3d_img_vit.json',
                                           override_json=True) 

    app()