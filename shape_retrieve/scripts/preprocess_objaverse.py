"""
This script iterates through the Objaverse dataset, extracts and stores features. 
"""
from shape_retrieve.data.allenai_objaverse import get_objaverse_all_model_ids, get_objaverse_model_path, get_objaverse_visual_path, get_objaverse_text_annotation
from shape_retrieve.DatasetInterface import DatasetInterface
from shape_retrieve.feature_extractor.vit import VIT_encoder
import hydra 
from omegaconf import DictConfig, OmegaConf

if __name__ == '__main__':
    @hydra.main(version_base=None, config_path="../../configs", config_name="data")
    def app(cfg: DictConfig):
        feature_extractor = VIT_encoder()

        # TODO make variable 
        N = 100000
        get_first_N = lambda cfg : {key: val[:N] if isinstance(val, list) else val for key,val in get_objaverse_all_model_ids (cfg).items()}
        dataset_interface = DatasetInterface(cfg,
                    get_model_path=get_objaverse_model_path,
                    get_visual_path=get_objaverse_visual_path,
                    get_text_annotation=get_objaverse_text_annotation,
                    get_all_model_ids=get_first_N)
        dataset_interface.extract_features(feature_extractor=feature_extractor,
                                            output_json=f'shape_retrieve/datasets/objaverse{N}_img_vit.json',
                                            override_json=True)

    app()