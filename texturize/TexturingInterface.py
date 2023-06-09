from texturize.TEXTurePaper.src.training.trainer import TEXTure
from texturize.TEXTurePaper.src.configs.train_config import TrainConfig # default config

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import random
from pathlib import Path
from loguru import logger
from hydra import compose, initialize


class TexturingInterface(object):
    def __init__(self, cfg: DictConfig,
                text: str = None, shape_path: str = None,
                backdrop_img: str = None,
                guidance_scale: float = None,
                seed: int = None):
        self.cfg = cfg

        self.texture_args = {
        'device': self.cfg.texturing.device,
        'log':{
            'exp_name': str(datetime.datetime.now()), # default
            'exp_root': None
         }, 'guide':{
            'text': None,
            'append_direction': None,
            'background_img': self.cfg.backdrop.default_path,
            'guidance_scale': 7.5,
            'shape_path': None,
         }, 'optim': {
            'seed': None,
         }}

        self.texture_args['guide']['append_direction'] = self.cfg.texturing.append_direction 
        self.texture_args['log']['exp_root'] = self.cfg.texturing.output_directory
        
        if text is not None:
            self.attach_text(text)
        if shape_path is not None:
            self.attach_shape_path(shape_path)
        if backdrop_img is not None:
            self.attach_backdrop_path(backdrop_img)
        if guidance_scale is not None:
            self.set_guidance_scale(guidance_scale)            
        if seed is None:
            seed = random.randint(0, 10000)
        self.set_seed(seed)
        

    def attach_text(self, text:str):
        if self.cfg.texturing.append_direction:
            self.texture_args['guide']['text'] = text + ', {} view'
        else:
            self.texture_args['guide']['text'] = text

    def attach_shape_path(self, shape_path: str):
        if not os.path.exists(shape_path) :
            raise ValueError('invalid shape path : {}'.format(shape_path))
        self.texture_args['guide']['shape_path'] = shape_path

    def attach_backdrop_path(self, backdrop_path: str):
        if backdrop_path is not None:
            if not os.path.exists(backdrop_path) :
                raise ValueError('invalid backdrop path : {}'.format(backdrop_path))
            self.texture_args['guide']['background_img'] = backdrop_path 

    def set_guidance_scale(self, scale:float):
        if scale is not None:
            self.texture_args['guide']['guidance_scale'] = scale

    def set_exp_root(self, root: str):
        self.texture_args['log']['exp_root'] = root

    def set_seed(self, seed: int):
        self.texture_args['optim']['seed'] = seed

    def set_name(self, name:str):
        self.texture_args['log']['exp_name']= name

    def run(self):
        # cfg = pyrallis.argparsing.ArgumentParser(config_class=TrainConfig)
        # import pyrallis # config loader used by TEXTure
        # cfg = pyrallis.parse(config_class=TrainConfig,)
        
        # initialize(version_base=None, config_path='../configs', job_name="texturing")
        cfg = compose(config_name="default_TEXTure_config") 
        cfg.log.exp_name = self.texture_args['log']['exp_name']
        cfg.log.exp_root = Path(self.texture_args['log']['exp_root'])
        cfg.guide.text = self.texture_args['guide']['text']
        cfg.guide.shape_path = self.texture_args['guide']['shape_path']
        cfg.guide.append_direction = self.texture_args['guide']['append_direction']
        cfg.optim.seed = self.texture_args['optim']['seed']
        cfg.guide.background_img = self.texture_args['guide']['background_img'] # os.path.join('texturize/TEXTurePaper', cfg.guide.background_img) 
        cfg.guide.guidance_scale = self.texture_args['guide']['guidance_scale']
        cfg.device = self.texture_args['device']
        
        cfg.log.exp_root = Path(cfg.log.exp_root)
        texture_module = TEXTure(cfg)
        texture_module.paint()



if __name__=='__main__':

    # Huggingface login
    from huggingface_hub import login
    with open('credentials/huggingface_key', 'r') as f:
        huggingface_token = f.readline().strip()
        login(token=huggingface_token)

    @hydra.main(version_base=None, config_path="../configs", config_name="data") 
    def app(cfg: DictConfig):
        interface = TexturingInterface(cfg)
        # interface.attach_shape_path('/orion/u/ianhuang/tabla_rasa/ScenesFromAbstractLanguage/texture/samples/presentation/objects/a_vase/a_vase_0.obj')
        # interface.attach_text('A jade vase with chinese patterns')
        # interface.set_name('jade vase')
        # interface.set_seed(3)
        # interface.run()
        
        query = cfg.texturing.query_instance if cfg.texturing.query_instance != str(None) else None
        shape_path = cfg.texturing.shape_path_instance if cfg.texturing.shape_path_instance != str(None) else None
        backdrop = cfg.texturing.backdrop_instance if cfg.texturing.backdrop_instance != str(None) else None
        guidance_scale = cfg.texturing.guidance_scale_instance if cfg.texturing.guidance_scale_instance != str(None) else None
        object_id = cfg.texturing.object_id_instance if cfg.texturing.object_id_instance != str(None) else None

        logger.info(f"query = {str(query)}")
        logger.info(f"shape_path = {str(shape_path)}")
        logger.info(f"backdrop = {str(backdrop)}")
        logger.info(f"guidance_scale = {str(guidance_scale)}")
        logger.info(f"object_id = {str(object_id)}")

        
        interface.attach_text(query) # language instruction for appearance
        # setup for this query...
        interface.attach_shape_path(shape_path) # tell it which shape_path to attend to
        interface.attach_backdrop_path(backdrop)
        interface.set_guidance_scale(guidance_scale)
        interface.set_exp_root(cfg.server.mesh_download_root)
        interface.set_name(object_id)
        interface.set_seed(cfg.general.seed) 
        interface.run() 
        
    app()