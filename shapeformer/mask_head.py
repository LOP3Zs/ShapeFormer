import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry



from .position_encoding import PositionEmbeddingLearned
from .mlp import MLP
from .transformer import (
    TransformerDecoder, TransformerEncoder, 
    TransformerDecoderLayer, TransformerEncoderLayer
)
from .transformer_m2f import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
)


from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY

from .generate_modules import (
    VectorQuantizedVAE, 
    ConditionalVectorQuantizedVAE,
)
from torchvision import transforms
from .layers_util import (
    init_roi_feature_learner,
    prepare_gt,
    mask_inference,
    mask_loss,
    cls_loss,
)

import numpy as np
import copy
import cv2
import os


__all__= ["ShapeFormerv2"]


@ROI_MASK_HEAD_REGISTRY.register()
class ShapeFormerv2(nn.Module):
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, vis_period=0, shapeformer=None, num_classes=None, **kwargs):
        super().__init__()
        conv_dim = input_shape.channels
        self.shapeformer = shapeformer
        self.vis_period = vis_period
        self.emb_dim = self.shapeformer.EMB_DIM
        self.num_classes = num_classes
        self.n_kv_feat_conv_layers = self.shapeformer.KV_FEAT_CONV_LAYERS
        self.n_roi_embed_conv_layers = self.shapeformer.ROI_EMBED_CONV_LAYERS

        # RoI embeddings - key value feature map
        self.visible_key_value_model = init_roi_feature_learner(conv_dim, 
                                            n_layers=self.n_kv_feat_conv_layers, upsample=False)
        self.visible_roi_embed = init_roi_feature_learner(conv_dim, 
                                            n_layers=self.n_roi_embed_conv_layers, upsample=True)
        self.amodal_key_value_model = init_roi_feature_learner(conv_dim, 
                                            n_layers=self.n_kv_feat_conv_layers, upsample=False)
        self.amodal_roi_embed = init_roi_feature_learner(conv_dim, 
                                            n_layers=self.n_roi_embed_conv_layers, upsample=True)

        # pe -- can be use for both visible and amodal kv feat
        self.positional_encoding = PositionEmbeddingLearned(self.emb_dim//2)

        # VISIBLE 
        decoder_layer = TransformerDecoderLayer(d_model=self.emb_dim, nhead=self.shapeformer.N_HEADS, 
                                                normalize_before=False)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.shapeformer.N_VI_LAYERS)
        self.query_embed = nn.Embedding(num_embeddings=2, embedding_dim=self.emb_dim)
        self.vis_mask_embed = MLP(self.emb_dim, self.emb_dim, self.emb_dim, 3)
        self.vis_class_embed = MLP(self.emb_dim, self.emb_dim, self.num_classes, 3)

        self.vi_predictor = init_roi_feature_learner(1, n_layers=0, upsample=False)
        self.bo_predictor = init_roi_feature_learner(1, n_layers=0, upsample=False)

        # SHAPE PRIOR SEARCHER
        self.use_sp_attn = self.shapeformer.SHAPEPRIOR_ATTENTION
        if self.use_sp_attn:
            self.prior_searcher = ConditionalVectorQuantizedVAE(
                input_dim=1,
                dim=self.emb_dim,
                n_conditions=self.num_classes,
                K=self.shapeformer.N_LATENT_VECTORS
            )
            if self.shapeformer.SEARCHER_PRETRAINED is not None:
                try:
                    self.prior_searcher.load_state_dict(torch.load(self.shapeformer.SEARCHER_PRETRAINED))
                    print("Successfully loaded the searcjer pretrained\
                            weights from {}".format(self.shapeformer.SEARCHER_PRETRAINED))
                except:
                    print("Failed to load the searcher pretrained from {}".format(self.shapeformer.SEARCHER_PRETRAINED))
                    print("Exiting...")
                    exit(0)


        # AMODAL
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.decoder_norm = nn.LayerNorm(self.emb_dim)
        self.vis_to_amodal_model = MLP(self.emb_dim, self.emb_dim, self.emb_dim, 3)
        self.a_mask_embed = MLP(self.emb_dim, self.emb_dim, self.emb_dim, 3)
        self.a_predictor = init_roi_feature_learner(1, n_layers=0, upsample=False)

        pre_norm = False
        for _ in range(self.shapeformer.N_A_LAYERS):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=self.emb_dim,
                    nhead=self.shapeformer.N_HEADS,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=self.emb_dim,
                    nhead=self.shapeformer.N_HEADS,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=self.emb_dim,
                    dim_feedforward=self.emb_dim,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        shapeformer = cfg.SHAPEFORMER
        vis_period = cfg.VIS_PERIOD

        ret.update(
            input_shape=input_shape,
            shapeformer=shapeformer,
            vis_period=vis_period
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        return ret

    def layers(self, x, inst_classes=None):
        x_ori = x.clone()
        bs = x_ori.shape[0]

        # VISIBLE
        visible_kv_feat = self.visible_key_value_model(x)
        visible_roi_embedding = self.visible_roi_embed(visible_kv_feat)

        pos_embed = self.positional_encoding.forward_tensor(visible_kv_feat)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        visible_kv_feat = visible_kv_feat.flatten(2).permute(2, 0, 1)
        decoder_output = self.transformer_decoder(tgt, visible_kv_feat, 
                                        pos=pos_embed,
                                        query_pos=None) # (1, n_masks, bs, dim)

        mask_embs = self.vis_mask_embed(decoder_output.squeeze(0).moveaxis(1,0))
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embs, visible_roi_embedding)
        vi_masks = self.vi_predictor(outputs_mask[:,0,:,:].unsqueeze(1)) #visible mask
        bo_masks = self.bo_predictor(outputs_mask[:,1,:,:].unsqueeze(1)) #occluder (bo - background objects) mask

        # SHAPE PRIOR SEARCHER
        class_logits = None
        if self.training:
            storage = get_event_storage()
            self.use_sp_attn = self.use_sp_attn and storage.iter > self.shapeformer.SP_START_ITER
        if self.use_sp_attn:
            class_logits = self.vis_class_embed(decoder_output.squeeze(0))[0]
            search_condition = inst_classes # class_logits.argmax(-1)
            shape_prior, _, _ = self.prior_searcher(vi_masks.tanh(), search_condition)
            shape_prior = F.interpolate(shape_prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
            shape_prior_attn_mask = (shape_prior.sigmoid().flatten(2).unsqueeze(1).\
                                        repeat(1, self.shapeformer.N_HEADS, 2, 1).flatten(0, 1) < 0.5).bool() # (bs*n_heads,n_target,n_source)
            # this is to ensure that when the shape prior is full True, it will not be used
            shape_prior_attn_mask[torch.where(shape_prior_attn_mask.sum(-1) == shape_prior_attn_mask.shape[-1])] = False
        else:
            shape_prior_attn_mask = None

        # AMODAL
        amodal_kv_feat = self.amodal_key_value_model(x)
        amodal_roi_embedding = self.amodal_roi_embed(amodal_kv_feat)

        a_pos_embed = self.positional_encoding.forward_tensor(amodal_kv_feat)
        a_pos_embed = a_pos_embed.flatten(2).permute(2, 0, 1)
        amodal_kv_feat = amodal_kv_feat.flatten(2).permute(2, 0, 1)

        a_decoder_output = self.vis_to_amodal_model(decoder_output.squeeze(0))

        for i in range(self.shapeformer.N_A_LAYERS):
            # attention: cross-attention first
            a_decoder_output = self.transformer_cross_attention_layers[i](
                a_decoder_output, amodal_kv_feat,
                memory_mask=shape_prior_attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=a_pos_embed, query_pos=None
            )


            a_decoder_output = self.transformer_self_attention_layers[i](
                a_decoder_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )
            
            # FFN
            a_decoder_output = self.transformer_ffn_layers[i](
                a_decoder_output
            )
        
        a_mask_embs = self.a_mask_embed(a_decoder_output[0])
        a_outputs_mask = torch.einsum("bc,bchw->bhw", a_mask_embs, amodal_roi_embedding)
        a_masks = self.a_predictor(a_outputs_mask.unsqueeze(1)) # amodal mask

        return vi_masks, bo_masks, a_masks, class_logits

    def forward(self, x, instances: List[Instances]):
        if self.training: # nn.Module attributes 
            inst_classes = cat([i.gt_classes for i in instances], dim=0)
            vi_masks, bo_masks, a_masks, class_logits = self.layers(x, inst_classes)

            loss_vi_mask = mask_loss(vi_masks, instances, mask_type="gt_visible_masks", vis_period =self.vis_period)
            loss_a_mask  = mask_loss(a_masks, instances, mask_type="gt_amodal_masks", vis_period =self.vis_period)
            loss_bo_mask = mask_loss(bo_masks, instances, mask_type="gt_background_objs_masks", vis_period =self.vis_period)
            loss_dict = {
                "loss_vi_mask": loss_vi_mask,
                "loss_a_mask": loss_a_mask,
                "loss_bo_mask": loss_bo_mask * self.shapeformer.BO_MASK_LOSS_WEIGHT,
            }

            if self.use_sp_attn:
                loss_mask_cls = cls_loss(class_logits, instances, vis_period=self.vis_period)
                loss_dict.update({"loss_mask_cls": loss_mask_cls})

            return loss_dict
        else:
            ## Inference forward 
            pred_inst_classes = cat([i.pred_classes for i in instances])
            vi_masks, bo_masks, a_masks, _ = self.layers(x, pred_inst_classes)
            mask_inference(vi_masks, instances, 'pred_visible_masks')
            mask_inference(bo_masks, instances, 'pred_occluding_masks')
            mask_inference(a_masks, instances, 'pred_amodal_masks')
            return instances
