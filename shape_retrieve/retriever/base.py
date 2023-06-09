
import torch

class GenericRetriever(object):
    def __init__(self):
        pass

    def evaluate(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def retrieve(self, x:list, anchors:dict, topk:int):
        # TODO: implement parallelism
        raise NotImplementedError