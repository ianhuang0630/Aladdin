import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

from shape_retrieve.feature_extractor.base import VisuoLinguisticEncoder        
    

class VIT_encoder(VisuoLinguisticEncoder):
    def __init__(self, device):
        super().__init__()
        assert device is not None
        self.device = device # 'cuda' if torch.cuda.is_available() else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
       
    def vit_large_patch14(self, img) -> torch.Tensor:
        with torch.no_grad():
            inputsImg = self.processor(images=img, return_tensors='pt').to(self.device)
            image_features = self.model.get_image_features(**inputsImg)
        return image_features

    def vit_text_embed(self, text: list[str]) -> torch.Tensor:
        assert all([isinstance(el, str) and len(el)>0 for el in text]), "need all valid strings!"
        with torch.no_grad():
            inputsText = self.processor(text=text, padding=True, return_tensors='pt').to(self.device)
            text_features = self.model.get_text_features(**inputsText)
        return text_features

    def vit_extract_text_img(self, batch: dict) -> list[dict] :
        text_valid = [(i, el) for i, el in enumerate(batch['text'])
                    if isinstance(el, str) and len(el) > 0]
        img_valid = [(i, el) for i, el in enumerate(batch['img'])
                    if el is not None]
       
        text_valid_idx = [el[0] for el in text_valid]
        img_valid_idx = [el[0] for el in img_valid]
        if len(text_valid_idx) > 0: 
            _text_feats = self.text_embed([el[1] for el in text_valid])
        if len(img_valid_idx) > 0:
            _img_feats = self.img_embed([el[1] for el in img_valid]) 

        text_feats = [None]*len(batch['text'])
        for idx, i in enumerate(text_valid_idx):
            text_feats[i] = _text_feats[idx].tolist()
        img_feats = [None]*len(batch['img'])
        for idx, i in enumerate(img_valid_idx):
            img_feats[i] = _img_feats[idx].tolist()

        features =  [{'img': img_feat, 'text': text_feat} for img_feat, text_feat in 
                    zip(img_feats, text_feats)] 
        return features 

    def text_embed(self, text: list[str]) -> torch.Tensor:
        return self.vit_text_embed(text)

    def img_embed(self, img) -> torch.Tensor:
        return self.vit_large_patch14(img)

    def embed(self, batch: dict) -> list[dict]:
        return self.vit_extract_text_img(batch)