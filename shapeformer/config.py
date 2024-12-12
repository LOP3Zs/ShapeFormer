from detectron2.config import CfgNode as CN

def add_shapeformer_config(cfg):
    # V2
    # 3kv - 1roi conv; 1 vi - 1 amo; wo attention
    cfg.SHAPEFORMER = CN()
    cfg.SHAPEFORMER.KV_FEAT_CONV_LAYERS = 3
    cfg.SHAPEFORMER.ROI_EMBED_CONV_LAYERS = 8

    cfg.SHAPEFORMER.N_HEADS = 2
    cfg.SHAPEFORMER.N_VI_LAYERS = 1
    cfg.SHAPEFORMER.N_A_LAYERS = 1
    cfg.SHAPEFORMER.EMB_DIM = 256

    cfg.SHAPEFORMER.SHAPEPRIOR_ATTENTION = True # if using attention on shape prior in amodal decoder
    cfg.SHAPEFORMER.SP_START_ITER = 40000 # number of steps start using shape prior
    cfg.SHAPEFORMER.USE_INST_CLASSES = True # if using gt classes as condition, not predicted classes
    cfg.SHAPEFORMER.SEARCHER_PRETRAINED = None # path to pretrained searcher weight
    cfg.SHAPEFORMER.N_LATENT_VECTORS = 128 # number of latent vectors of the searcher

    cfg.SHAPEFORMER.BO_MASK_LOSS_WEIGHT = 1.

