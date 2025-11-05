import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import numpy as np
import torch.nn.functional as F


'''hook function'''

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output




class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
     

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    '''token filter function for DAMRO'''
    def token_filter(self, images, top_k):
        
        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]
        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        #filter the top k token featrures
        cls_attn = attn[:, 0, 1:]            
        _, idx = torch.topk(cls_attn, top_k, dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
        image_features_filtered = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
       
        return image_features_filtered


    @torch.no_grad()
    def forward(self, images,top_k):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if top_k==None:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
            else:
                image_features=self.token_filter(images, top_k)
          

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
