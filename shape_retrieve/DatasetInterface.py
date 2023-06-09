"""
DatasetInterface
"""
import os
from typing import Callable
from loguru import logger
import datetime 
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from typing import Type

from torch.utils.data import DataLoader, Dataset

from PIL import Image
import requests
import multiprocessing
import copy
from io import BytesIO

from utils.io import file_location_type, read_img

from shape_retrieve.data.allenai_objaverse import get_objaverse_all_model_ids, get_objaverse_model_path, get_objaverse_visual_path, get_objaverse_text_annotation
from shape_retrieve.data.future3d import get_Future3D_all_model_ids, get_Future3D_model_path, get_Future3D_visual_path, get_Future3D_text_annotation
from shape_retrieve.feature_extractor.base import GenericEncoder, VisuoLinguisticEncoder
from shape_retrieve.feature_extractor.vit import VIT_encoder

from shape_retrieve.retriever.base import GenericRetriever
from shape_retrieve.retriever.cossine_sim import CossineSim


class img_dataset(Dataset):
    def __init__(self, 
                get_visual_path: Callable[[str], dict],
                get_text_annotation: Callable[[str], dict],
                get_all_model_ids: Callable[[], dict]):
        super().__init__() 
        self.get_visual_path = get_visual_path
        self.get_text_annotation =  get_text_annotation
        self.get_all_model_ids = get_all_model_ids
        id_info = self.get_all_model_ids()
        self.model_ids = id_info['ids']
        self.data_format = id_info['format']
    
    def __len__(self):
        return len(self.model_ids)
        
    def __getitem__(self, idx):
        model_id = self.model_ids[idx] 
        img = self.get_visual_path(model_id) # potential download happening here. 
        img_format = img['format']
        img_file = img['path']
        text_annot = self.get_text_annotation(model_id)['text']

        if img_format.endswith('-url'):
            assert file_location_type(img_file) == 'url'
        try:
            img = read_img(img_file)
        except:
            logger.info(f"error loading {img_file}")
            img = None 
            
        return {'id': model_id, 'img': img, 'text': text_annot, 'format': self.data_format}

class DatasetInterface(object):
    """
    Produces a .json : [model id] --> {'language': [language embedding], 'visual': [visual embedding]}

    Retrieves the path of every 3D model.

    """
    def __init__(self,
                 cfg: DictConfig, 
                 get_model_path: Callable[[DictConfig, str], dict],
                 get_visual_path: Callable[[DictConfig, str], dict],
                 get_text_annotation: Callable[[DictConfig, str], dict],
                 get_all_model_ids: Callable[[DictConfig], dict]
                 ):

        self.cfg = cfg 
        self._get_model_path = get_model_path
        self._get_visual_path = get_visual_path
        self._get_text_annotation = get_text_annotation
        self._get_all_model_ids = get_all_model_ids

        self.get_model_path = lambda x: get_model_path(cfg=cfg, id=x)
        self.get_visual_path = lambda x : get_visual_path(cfg=cfg, id=x)
        self.get_text_annotation = lambda x: get_text_annotation(cfg=cfg, id=x)
        self.get_all_model_ids = lambda : get_all_model_ids(cfg=cfg)

        self.format = self.get_all_model_ids()['format']

        # assigned on .extract_features()
        self._shape_ids = None
        self._shape_features_metadata = None
        self._shape_features = None

    @property 
    def shape_ids(self):
        if self._shape_ids is None: 
            raise ValueError("Have you run extract_features?") 
        else:
            return self._shape_ids

    @property 
    def shape_features(self):
        if self._shape_features is None:
            raise ValueError("Have you run extract_features?") 
        else:
            return self._shape_features

    @property
    def shape_features_metadata(self):
        if self._shape_features_metadata is None:
            raise ValueError("Have you run extract_features?")
        else:
            return self._shape_features_metadata

    def model_path(self, results: list[dict]) -> list[str]:
        return [self.get_model_path(el['id']) for el in results]

    def img_path(self, results: list[dict]) -> list[str]:
        return [self.get_visual_path(el['id']) for el in results]
        
    def text_annots(self, results: list[dict]) -> list[str]:
        return [self.get_text_annotation(el['id']) for el in results]



    def language_retrieve(self, query: list[str], 
                 topk: int, 
                 feature_extractor: Type[VisuoLinguisticEncoder],
                 retriever: Type[GenericRetriever],
                 **kwargs
                 ) -> list[dict]:
        """ Retrieves the top K
        Args:
            query: list of strings
            topk: the integers 
        """
        # encode the queries
        assert isinstance(feature_extractor, VisuoLinguisticEncoder)
        text_query = feature_extractor.text_embed(query)
        assert type(feature_extractor).__name__== self.shape_features_metadata['feature-extractor'], \
            "mismatch between the feature extractor given and the one used to generate the features"
        text_query = text_query.detach().cpu().tolist()

        anchors = self.shape_features
        assert isinstance(anchors, dict)
        anchors['id'] = self.shape_ids  # adding additional field needed by retriever
        anchors['format'] = [self.format]*len(self.shape_ids)
        results = retriever.retrieve(x=text_query, anchors=anchors,
                                     topk=topk, **kwargs)
        # list of length query, where each dictionary contains ID and score information
        return results

    def get_dataloader(self,):
        imgD = img_dataset(self.get_visual_path, self.get_text_annotation, self.get_all_model_ids) 
        dataloader = DataLoader(imgD, batch_size=self.cfg.processing.img_batchsize, 
                        num_workers= min(multiprocessing.cpu_count(), 6), 
                        collate_fn=lambda x: {key: [el[key] for el in x] for key in x[0]})
        return dataloader

    def extract_features(self,
                        feature_extractor: Type[GenericEncoder],
                        output_json: str,
                        override_json: bool = False):
        """ 
        Args:
            feature_extractor: a function that takes in a batch and returns a list of dictionaries
            output_json: The location of where to save.
            override_json: True if you would like it to ignore previously saved features, and re-generate from scratch.
        """
        if output_json is not None and not override_json and os.path.exists(output_json):
            logger.info(f'{output_json} found and override is switched off, loading precomputed features!')
            with open(output_json, 'r') as f:
                data = json.load(f)
            logger.info('Done!')
        else: 
            if output_json is not None:
                if os.path.exists(output_json):
                    logger.info(f'{output_json} found but override is on, recomputing features!')
                else:
                    logger.info(f'{output_json} is not found, computing features!')
             
            model_id2rep = {}
            dataloader = self.get_dataloader()  
            logger.info('starting feature extraction...')
            for batch in tqdm(dataloader):
                features = feature_extractor.embed(batch) # batch
                
                # now go through them:
                for idd, feat in zip(batch['id'], features):
                    if idd in model_id2rep:
                        logger.info(f"{idd} has repeated feature_extractor")
                        if feat != model_id2rep[idd]:
                            logger.info(f"feature extracted for {idd} does not match previous")
                            raise ValueError("Inconsistent feature extracted")
                    model_id2rep[idd] = feat# making it jsonifiable
            
            logger.info('done!')
            data = {
                'time-generated': str(datetime.datetime.now()),
                'model-path': self._get_model_path.__name__,
                'visual-path': self._get_visual_path.__name__,
                'feature-extractor': type(feature_extractor).__name__,
                'features': model_id2rep
            }     

            if output_json is not None:
                with open(output_json, 'w') as f:
                    json.dump(data, f )
                logger.info(f"saved at {output_json}")

        # feature post-processing
        self._shape_features = {}  
        self._shape_features_metadata = {key: data[key] for key in data if key not in ('features', )}
        self._shape_ids = []
        for id_ in data['features']:
            self._shape_ids.append(id_)
            for feature_name in data['features'][id_]:
                if feature_name not in self.shape_features:
                    self._shape_features[feature_name] = []
                self._shape_features[feature_name].append(data['features'][id_][feature_name])

        return self


class DatasetEnsembleInterface(object):
    def __init__(self,
                 cfg: DictConfig,
                 datasets: list[tuple[str, DatasetInterface]]
                ):
        """
        
        """
        assert all([isinstance(el, DatasetInterface) for el in datasets])
        self.datasets = datasets
        self.form2dataset = {d.format : d for d in self.datasets}

    @property
    def shape_ids(self):
        raise NotImplementedError

    @property
    def shape_features(self):
        raise NotImplementedError

    @property
    def shape_features_metadata(self):
        raise NotImplementedError

    def model_path(self, results: list[dict]) -> list[str]:
        out = []
        for res in results:
            out.extend(self.form2dataset[res['format']].model_path([res]))
        return out

    def img_path(self, results: list[dict]) -> list[str]:
        out = []
        for res in results:
            out.extend(self.form2dataset[res['format']].img_path([res]))
        return out

    def text_annots(self, results: list[dict]) -> list[str]:
        out = []
        for res in results:
            out.extend(self.form2dataset[res['format']].text_annots([res]))
        return out

    def language_retrieve(self, query: list[str], 
                 topk: int, 
                 feature_extractor: Type[VisuoLinguisticEncoder],
                 retriever: Type[GenericRetriever],
                 **kwargs
                 ) -> list[dict]:
        """ Retrieves the top K
        Args:
            query: list of strings
            topk: the integers 
        """
        results = [None]*len(query) # len = n_queries
        for d_idx, dataset in enumerate(self.datasets):
            retrieved_objs = dataset.language_retrieve(query, topk, feature_extractor, 
                                                       copy.deepcopy(retriever), **kwargs)
            for o_idx, retrieved_obj in enumerate(retrieved_objs):
                if results[o_idx] is None:
                    results[o_idx] = []
                results[o_idx].append(retrieved_obj)
        pruned_options = [] 

        # iterating across n_objects
        for perobj_results in zip(results):
            options = [] 
            for el in perobj_results[0]:
                options.extend(el)
            options = [(el, el["sim-score"]) for el in options]
            options = sorted(options, key=lambda x: x[1], reverse=True)
            options = options[:min(len(options), topk)]
            options = [el[0] for el in options]
            pruned_options.append(options)
            
        return pruned_options


if __name__ == '__main__':
    
    @hydra.main(version_base=None, config_path="../configs", config_name="data")
    def app(cfg: DictConfig):
        feature_extractor = VIT_encoder()
        retriever = CossineSim()

        # objaverse 30k
        get_first_30k = lambda cfg : {key: val[:30000] if isinstance(val, list) else val for key,val in get_objaverse_all_model_ids (cfg).items()}
        objaverse_dataset_interface = DatasetInterface(cfg,
                    get_model_path=get_objaverse_model_path,
                    get_visual_path=get_objaverse_visual_path,
                    get_text_annotation=get_objaverse_text_annotation,
                    get_all_model_ids=get_first_30k)
        objaverse_dataset_interface.extract_features(feature_extractor=feature_extractor,
                                          output_json='shape_retrieve/datasets/objaverse30000_img_vit.json',
                                          override_json=False)

        future3d_dataset_interface = DatasetInterface(cfg,
                    get_model_path=get_Future3D_model_path,
                    get_visual_path=get_Future3D_visual_path,
                    get_text_annotation=get_Future3D_text_annotation,
                    get_all_model_ids=get_Future3D_all_model_ids)    
        future3d_dataset_interface.extract_features(feature_extractor=feature_extractor,
                                          output_json='shape_retrieve/datasets/future3d_img_vit.json', 
                                          override_json=False)

        ensemble_interface = DatasetEnsembleInterface(cfg, [future3d_dataset_interface, objaverse_dataset_interface])
        
        results = ensemble_interface.language_retrieve([
            'an old car',
            'a tall tree',
            'a banana',
        ], topk=3, retriever=retriever, feature_extractor=feature_extractor)

        for res in results:
            print(res)
            annots = ensemble_interface.text_annots(res)
            print(annots)



        # dataset_interface = DatasetInterface(cfg, 
        #             get_model_path=get_Future3D_model_path,
        #             get_visual_path=get_Future3D_visual_path,
        #             get_text_annotation=get_Future3D_text_annotation,
        #             get_all_model_ids=get_Future3D_all_model_ids)
        
        # dataset_interface.extract_features(feature_extractor=feature_extractor,
        #                                    output_json='shape_retrieve/datasets/future3d_img_vit.json',
        #                                    override_json=False) 

        # results = dataset_interface.language_retrieve([
        #     'a tall wooden chair',
        #     'a square modern table',
        #     'chinese lamp',
        # ], topk=3, retriever=retriever, feature_extractor=feature_extractor)
        # for res in results:
        #     annots =  dataset_interface.text_annots(res)
        #     print(annots)
            

    app()
