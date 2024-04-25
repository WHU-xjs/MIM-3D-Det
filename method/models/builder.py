from mmcv.utils import Registry
from mmdet3d.models.builder import BACKBONES, HEADS, LOSSES, NECKS, MODELS
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
from mmdet.models.builder import NECKS as MMDET_NECKS
from mmdet.models.builder import HEADS as MMDET_HEADS

FUSERS = Registry("fusion_users")
ENCODERS = Registry("encoders")
DECODERS = Registry("decoders")
ATTENTIONS = Registry("attentions")
CONV_GROUP = Registry("conv_group")
VTRANSFORMS = Registry("vtransforms")

def build_attention(cfg):
    return ATTENTIONS.build(cfg)

def build_backbone(cfg):
    if cfg['type'] in BACKBONES._module_dict.keys():
        return BACKBONES.build(cfg)
    else:
        return MMDET_BACKBONES.build(cfg)

def build_neck(cfg):
    if cfg['type'] in NECKS._module_dict.keys():
        return NECKS.build(cfg)
    else:
        return MMDET_NECKS.build(cfg)

def build_head(cfg):
    if cfg['type'] in HEADS._module_dict.keys():
        return HEADS.build(cfg)
    else:
        return MMDET_HEADS.build(cfg)

def build_loss(cfg):
    return LOSSES.build(cfg)

def build_fuser(cfg):
    return FUSERS.build(cfg)

def build_encoder(cfg):
    return ENCODERS.build(cfg)

def build_decoder(cfg):
    return DECODERS.build(cfg)

def build_conv_group(cfg):
    return CONV_GROUP.build(cfg)

def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)

def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function warpper for building 3D detector or segmentor according to
    cfg.

    Should be deprecated in the future.
    """
    return MODELS.build(cfg)