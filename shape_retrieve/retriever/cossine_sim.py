

"""
Implementation of cosine similarity retrieval
"""
import torch
from shape_retrieve.retriever.base import GenericRetriever
from loguru import logger

class CossineSim(GenericRetriever):
    def __init__(self, text_weight: float = 0.5, device=None):
        """
        Args:
            threshold: hard threshold for cossine similarity. Samples that fall below 
                       this threshold will not be included.
            text_weight: weight for text-based similarity, if text info available.
            image_weight: weight for vision-based similarity
        """        
        super().__init__()

        self.weighting = {'text': text_weight}
        
        assert device is not None
        self.device = device #'cuda' if torch.cuda.is_available() else 'cpu'


        # caching, for reuse of this retrieval object
        self._valid_text = None
        self._text_valid_indices = None
        self._valid_img = None
        self._img_valid_indices = None

    def evaluate(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            x: the query features (B x D)
            anchors: the features of candidates (N x D)
        """ 

        assert len(x.shape) == 2 and len(anchors.shape) == 2
        assert x.shape[-1] == anchors.shape[-1]
        x_normed = x/x.norm(dim=-1, keepdim=True) 
        anchors_normed = anchors/anchors.norm(dim=-1, keepdim=True)
        cossine_sim = torch.matmul(x_normed, anchors_normed.transpose(0, 1))
        return cossine_sim

    def retrieve(self, x:list, anchors:dict, topk:int, threshold:float=None) -> list[dict]:

        """
        Args:
            x: list ( length B ) with the query features
            anchors: dictionary, where keys are feature names, and keys are matrices
            topk: the top K to return
        """ 
        if 'img' not in anchors:
            raise ValueError('provided features do not have img field')
        if 'text' not in anchors:
            raise ValueError('provided features do not have text field')

        x = torch.tensor(x).to(self.device)
        ids = anchors['id']
        form = anchors['format']

        # check if the text embedding is available. 
        if self._valid_text is None and self._text_valid_indices is None:
            logger.info("initializing text embeddings and valid text indices...")
            valid_text = [el for el in anchors['text'] if isinstance(el, list)]
            text_valid_indices = [i for i in range(len(anchors['text'])) 
                                    if isinstance(anchors['text'][i], list)]
            self._valid_text = valid_text
            self._text_valid_indices = text_valid_indices
            logger.info("Done.")
        else:
            logger.info("Using cached text embeddings and valid text indices.")
            valid_text = self._valid_text
            text_valid_indices = self._text_valid_indices
        anchors_text = None
        if len(valid_text) > 0:
            anchors_text = torch.tensor(valid_text).to(self.device)

        # check if the image embedding is available
        if self._valid_img is None and self._img_valid_indices is None:
            logger.info("initializing image embeddings and valid image indices...")
            valid_img = [el for el in anchors['img'] if isinstance(el, list)]
            img_valid_indices = [i for i in range(len(anchors['img']))
                                    if isinstance(anchors['img'][i], list)] 
            self._valid_img = valid_img
            self._img_valid_indices = img_valid_indices
            logger.info("Done.")
        else:
            logger.info("Using cached image embeddings and valid image indices.")
            valid_img = self._valid_img
            img_valid_indices = self._img_valid_indices
        anchors_img = None 
        if len(valid_img) > 0:
            anchors_img = torch.tensor(valid_img).to(self.device)

        # cossine sim evaluation 
        sim_img = None
        if anchors_img is not None:
            sim_img = self.evaluate(x, anchors_img)

        sim_text = None
        if anchors_text is not None: 
            sim_text = self.evaluate(x, anchors_text)
        if sim_img is None and sim_text is None:
            raise ValueError("Dataset does not have any valid embeddings!")

        # evaluation of the final similarities
        if sim_img is None:
            sim_agg = sim_img
        if sim_text is None:
            sim_agg = sim_text
        else:
            # each weight factor is x.shape[0], len(ids)
            text_valid = torch.zeros(x.shape[0], len(ids), dtype=torch.bool).to(self.device)
            text_valid[:, text_valid_indices] = True # self.weighting['text']
            # original sim_text with padding
            _sim_text = torch.zeros_like(text_valid, dtype=torch.float).to(self.device)
            _sim_text[:, text_valid_indices] = sim_text
            sim_text = _sim_text

            # each weight factor is x.shape[0], len(ids)
            img_valid = torch.zeros(x.shape[0], len(ids), dtype=torch.bool).to(self.device)
            img_valid[:, img_valid_indices] = True
            # original sim_img with padding
            _sim_img = torch.zeros_like(img_valid, dtype=torch.float).to(self.device)
            _sim_img[:, img_valid_indices] = sim_img
            sim_img = _sim_img

            # NOTE: 
            # img_weight and text_weight can be set to 0 when ~img_valid 
            # img_weight = (1-self.weighting['text']) when img_valid AND text_valid

            # last round of filtering, depending on preferences 
            # valid_entry = torch.logical_or(img_valid[0], text_valid[0])
            valid_entry = img_valid[0]
            

            img_weight = (
                (1-self.weighting['text']) * torch.logical_and(img_valid, text_valid).float()
                +
                torch.logical_and(img_valid, torch.logical_not(text_valid)).float()
            )
            text_weight = (
                self.weighting['text'] * torch.logical_and(img_valid, text_valid).float()
                +
                torch.logical_and(torch.logical_not(img_valid), text_valid).float()
            )
                        
            # final aggregated similarity
            sim_agg = text_weight*sim_text + img_weight*sim_img
            sim_agg[:, torch.logical_not(valid_entry )] = -1

        topk_sim = torch.topk(sim_agg, topk, dim=-1, largest=True)
        topk_scores = topk_sim.values
        topk_indices = topk_sim.indices
        
        # check valid (above certain threshold) 
        if threshold:
            valid = topk_scores >= threshold
        else:
            valid = torch.ones_like(topk_scores)
        
        # wrapup.
        results = [] 
        for v_row, scores_row, indices_row in zip(valid, topk_scores, topk_indices):
            this_sample_results = []
            for v, score, index in zip(v_row, scores_row, indices_row):
                if v: # if valid
                    this_sample_results.append({'id': ids[index], 
                                                'sim-score': score.detach().cpu().item(),
                                                'format': form[index]})
            results.append(this_sample_results) 
        return results
