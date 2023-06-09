"""
Implementation fo CLIP-Similarity metric
"""
import torch
from typing import Type, Union, Tuple
from utils.records import * 
from shape_retrieve.feature_extractor import vit
from PIL import Image, PngImagePlugin
import numpy as np


class CLIPBasedMetric(object):
    def __init__(self, device):
        self.device = device
        self.clip_encoder = vit.VIT_encoder(self.device)

    def prepare_thumbnails_and_images (self, session_token: str, event_dir='interaction_records/') -> Tuple[list[list[Type[PngImagePlugin.PngImageFile]]], list[list[Type[PngImagePlugin.PngImageFile]]]]:
        """  
        """
        # EventCollection().load_from_session(event_dir, session_token).filter_to('')
        texturing_IO = EventCollection().load_from_session(event_dir, session_token).get_texturing_timeline()
        renderings_per_output = []
        thumbnails_per_output = []
        for transaction in texturing_IO:
            request_event = transaction.input_event
            texture_event = transaction.output_event
            if not os.path.exists(texture_event.textured_output_paths['top_directory']):
                continue
            img_root_path = os.path.join(texture_event.textured_output_paths['top_directory'], 'vis',  'eval')
            files = os.listdir(img_root_path)

            # gtting the renderings
            if len(files) == 0:
                 continue
            max_paintstep = max([int(el.split('_')[1]) for el in files if el.startswith('step') and el.endswith('.jpg')])
            images_files =  [el for el in files if el.startswith('step') and el.endswith('.jpg') and int(el.split('_')[1]) == max_paintstep]


            # getting the thumbnaisl    

            try:
                thumbnail_files = [read_img(request_event.shape_info['image'])]
            except:
                continue
            thumbnails_per_output.append(thumbnail_files) 

            pil_images_files = []
            for el  in images_files:
                try:
                    pil_images_files.append(read_img(os.path.join(img_root_path, el)))
                except:
                    continue
            if len(pil_images_files)== 0:
                continue
            renderings_per_output.append(pil_images_files)

        assert len(renderings_per_output) == len(thumbnails_per_output)
        return (thumbnails_per_output, renderings_per_output)
            

        # load the  models  that are retrieved
        # load the models that are textured, and then go back to try to get the images from the part that is retrieved
 
    def prepare_images(self, session_token: str, event_dir='interaction_records/') ->  list[list[Type[PngImagePlugin.PngImageFile]]]:
        textured_outputs = EventCollection().load_from_session(event_dir, session_token).time_sorted().filter_to('TEXTURED_OBJECT')

        renderings_per_output = []
        for texture_event in textured_outputs:
            if not os.path.exists(texture_event.textured_output_paths['top_directory']):
                continue
        
            img_root_path = os.path.join(texture_event.textured_output_paths['top_directory'], 'vis',  'eval')
            files = os.listdir(img_root_path)

            if len(files) == 0:
                 continue
            max_paintstep = max([int(el.split('_')[1]) for el in files if el.startswith('step') and el.endswith('.jpg')])
            images_files =  [el for el in files if el.startswith('step') and el.endswith('.jpg') and int(el.split('_')[1]) == max_paintstep]
            
            pil_images_files = []
            for el  in images_files:
                try:
                    pil_images_files.append(read_img(os.path.join(img_root_path, el)))
                except:
                    continue
            if len(pil_images_files)== 0:
                continue
            renderings_per_output.append(pil_images_files)
            
        return renderings_per_output  

    
    def prepare_texts(self, session_token: str, event_dir='interaction_records/') -> list[str]:
        # first, if the  session_token happens to end with '_baseline', then actually the text description is inthe ORIGINAL folder.
        if session_token.endswith('_baseline'):
            session_token = session_token[:-len('_baseline')]
        scene_description = EventCollection().load_from_session(event_dir, session_token).filter_to('INPUT_SCENE_DESCRIPTION')[-1].scene_description
         
        # TODO : allow different text prompt templates
        match1 = f'an element in a scene of {scene_description}'
        match2 = f'an object from a scene of {scene_description}'
        match3 = f'a picture of an object from {scene_description}'
        match4 = f'a rendering of an asset from a 3D scene of {scene_description}'
        match5 = f'{scene_description}'
        return [match1, match2, match3, match4, match5]

    def normalize_vec(self, x: Type[torch.tensor]) -> Type[torch.Tensor]:
        x_normed = x/x.norm(dim=-1, keepdim=True) 
        return x_normed

    def eval_cossine_sim(self, x: Type[torch.tensor], anchors: Type[torch.Tensor]) -> Type[torch.Tensor]:
        """ Evaluates the mean CLIP Similarity for embeddings 
        """
        assert len(x.shape) == 2 and len(anchors.shape) == 2
        assert x.shape[-1] == anchors.shape[-1]
        x_normed = self.normalize_vec(x)
        anchors_normed = self.normalize_vec(anchors)
        cossine_sim = torch.matmul(x_normed, anchors_normed.transpose(0, 1))
        return cossine_sim


    def evaluate_on_session(self, session_token: str, event_dir='interaction_records'):
        raise NotImplementedError
        
# clip diversity  (negative mean pairwise cossine similarity)
class CLIPDiversity(CLIPBasedMetric):
    def __init__(self, device):
        super().__init__(device)

    def evaluate_on_session(self, session_token: str, event_dir='interaction_records'):
        img_input = self.prepare_images(session_token, event_dir)
        img_emb = [] 
        for views in img_input:
            view_embedding = self.clip_encoder.img_embed(views) 
            view_embedding = self.normalize_vec( view_embedding )
            view_embedding = torch.mean(view_embedding, keepdim=True, dim=0)
            img_emb.append(view_embedding)
        img_emb = torch.cat(img_emb, dim=0)

        cossine_sim = self.eval_cossine_sim(img_emb, img_emb)
        upper_triangular_indices  = torch.triu_indices(cossine_sim.shape[0], cossine_sim.shape[1], offset=1, device=self.device) 
        # TODO protect against case when only a single object is there -- diversity term  is  ill-defined

        # TODO: below is  shit code. there  must be a better way.
        diversity_terms =[]
        for indices in upper_triangular_indices.transpose(0,1):
            diversity_terms.append(- cossine_sim[indices[0].item(), indices[1].item ()].detach().cpu())

        diversity = np.mean(diversity_terms)    
    
        return diversity
   

    
# clip sim for ablation between textured vs non-textured
 

# TODO come  up with a metric base-class, if this is something that should be scaled up
# to many metric suites.
class CLIPSimilarity(CLIPBasedMetric):
    def __init__(self, device):
        super().__init__(device)

    def evaluate_cross_session(self, session_tokens: list[str], event_dir='interaction_records'):
        texts_per_session = []        
        for session_token in session_tokens:
            texts = self.prepare_texts(session_token, event_dir=event_dir)
            texts_per_session.append(texts) 
        variations_per_session = len(texts_per_session[0])
        num_sessions = len(texts_per_session) 
        
        texts_per_session_flattened = []
        for i in range (num_sessions):
            for j in range(variations_per_session):
                texts_per_session_flattened.append(texts_per_session[i][j])

        session_sims = [] 

        correct_prediction = []

        for correct_idx, session_token in enumerate(session_tokens): 
            this_session_renderings = self.prepare_images(session_token, event_dir=event_dir)
            this_session_sim = self.run(this_session_renderings, texts_per_session_flattened, aggregation=None) # 

            # for i in range(num_sessions):
            # this_session_renderings = renderings_per_session[i ]
            this_session_sim = this_session_sim.reshape(-1, num_sessions, variations_per_session)
            this_session_sim = this_session_sim.max(-1).values #  max over all the different views
            # now we classify
            this_session_correct_classification = (torch.argmax(this_session_sim, dim=-1) == correct_idx)
            correct_prediction.append(this_session_correct_classification) 
            session_sims.append(this_session_sim)

        correct_prediction = torch.cat(correct_prediction)
        session_sims = torch.cat(session_sims,  dim=0)
        return session_sims, correct_prediction


    def score_cross_sim(self, session_sims):
        
        correct_sim = (torch.eye(session_sims.shape[0], session_sims.shape[1]).unsqueeze(-1).to(self.device) *  session_sims).sum(1)
        total_sim = session_sims.sum(1)

        return correct_sim/total_sim
        

    def evaluate_on_session(self, session_token: str, event_dir='interaction_records'):
        """ Evaluates the mean CLIP  Similarity to different text prompts for a given session_token
        """
        img_input = self.prepare_images(session_token, event_dir=event_dir)
        text_input =  self.prepare_texts(session_token, event_dir=event_dir)
        cossine_sim = self.run(img_input, text_input)
        return cossine_sim

    def evaluate_on_session_thumbnail_vs_textured(self, session_token: str, event_dir='interaction_records/'): 
        thumb_input, img_input = self.prepare_thumbnails_and_images(session_token, event_dir=event_dir)
        text_input = self.prepare_texts(session_token, event_dir=event_dir)
        img_cossine_sim = self.run(img_input, text_input, aggregation=None) # setting aggregation=None  returns the whole per-item sim scores
        thumb_cossine_sim = self.run(thumb_input, text_input, aggregation=None )
        return thumb_cossine_sim, img_cossine_sim


    def run(self, 
            images: list[list[Type[PngImagePlugin.PngImageFile]]], 
            text: Union[list[str], str],
            aggregation: 'str' = 'mean'
            ):
        if isinstance(text, str):
            text = [text]

        text_embedding = self.clip_encoder.text_embed(text) 

        sim_per_obj_per_text = [] 
        for object_renderings in images:
            images_embedding = self.clip_encoder.img_embed(object_renderings)
            cossine_sim = self.eval_cossine_sim(images_embedding, text_embedding)
            
            # cossine_sim = num_renderings x num_texts
            cossine_sim = torch.max(cossine_sim, dim=0, keepdim=True).values # 1 x num_texts
            sim_per_obj_per_text.append(cossine_sim)
        sim_per_obj_per_text = torch.cat(sim_per_obj_per_text, dim=0)
        
        if  aggregation is None : 
            final_sim = sim_per_obj_per_text
        elif aggregation == 'mean':
            final_sim = torch.mean(sim_per_obj_per_text, dim = 0)
        else:
            raise NotImplementedError( f'{aggregation} is not an implemented aggregation method so far.')
        return final_sim
            
        

if __name__ == '__main__':

    ours_sessions = [] #['western_saloon', 'rustic_backyard', 'confucius_bedroom']
    baseline_sessions = [] #['western_saloon_baseline',  'rustic_backyard_baseline', 'confucius_bedroom_baseline']
    exclude = [] # ['antichrist_vatican_baseline', 'mad_scientist_restaurant_baseline']
    print('analysis covering...')
    for idx,  sess in enumerate([el for el in os.listdir('interaction_records/') if el.endswith('_baseline')]):
        print(f'{idx+1} : {sess[:-len("_baseline")]}')
        if sess in exclude:
            continue
        ours_sessions.append(sess[:-len("_baseline")])
        baseline_sessions.append(sess)
     
    CLIPS = CLIPSimilarity("cuda:0")
    CLIPD = CLIPDiversity("cuda:0") 

    print('################### cross-sample analysis ############################')

    ours_obj2scene_sims,  ours_obj2scene_results = CLIPS.evaluate_cross_session(ours_sessions)
    print(f'(ours) Num assets generated : {ours_obj2scene_sims.shape}')
    print(f'(ours) Object2Scene prediction accuracy: {np.round(torch.mean(ours_obj2scene_results.float()).detach().cpu().numpy()*100, 2)}%')

    baseline_obj2scene_sims, baseline_obj2scene_results = CLIPS.evaluate_cross_session(baseline_sessions)
    print(f'(baseline) Num assets generated: {baseline_obj2scene_sims.shape}')
    print(f'(baseline) Object2Scene prediction accuracy: {np.round(torch.mean(baseline_obj2scene_results.float()).detach().cpu().numpy()*100, 2)}%')   

    print('################### per-sample analysis ##############################')

    for ours, baseline in zip(ours_sessions, baseline_sessions):
        print('#'*18) 

        diversity_ours = CLIPD.evaluate_on_session(ours)
        diversity_baseline = CLIPD.evaluate_on_session(baseline)

        print(f'{ours}')
        print(f'Diversity : {diversity_ours} (Ours) v {diversity_baseline} (Baseline)')

        similarity_ours = torch.mean(CLIPS.evaluate_on_session(ours)).detach().cpu().item()
        similarity_baseline = torch.mean(CLIPS.evaluate_on_session(baseline)).detach().cpu().item()
        print(f'Similarity: {similarity_ours} (Ours) v {similarity_baseline} (Baseline)')
        print(f'Beat baseline? {(diversity_ours + similarity_ours) > (diversity_baseline + similarity_baseline)}')

        
        thumbnail_sim, textured_sim = CLIPS.evaluate_on_session_thumbnail_vs_textured(ours)     
        print(f"{np.round(torch.mean((thumbnail_sim.mean(-1) < textured_sim.mean(-1)).float()).detach().cpu().item() *100, 2)}% of outputs score higher CLIP Similarity when textured.")
        

