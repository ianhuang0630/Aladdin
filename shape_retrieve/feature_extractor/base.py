import torch 

class GenericEncoder(object):
    def __init__(self):
        pass
    
    def embed(self, batch: dict):
        raise NotImplementedError

class VisuoLinguisticEncoder(GenericEncoder):
    def __init__(self):
        super().__init__()
    
    def embed(self, batch: dict) -> list[dict]:
        raise NotImplementedError

    def text_embed(self, text: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def img_embed(self, img) -> torch.Tensor:
        raise NotImplementedError


