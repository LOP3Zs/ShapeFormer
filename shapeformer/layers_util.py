import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
import fvcore.nn.weight_init as weight_init
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage


def init_roi_feature_learner(in_dim, n_layers, upsample=True, out_dim=None):
    '''
    this module enrich the CxHxW roi feature with convolutional layers
    the obtained feature will be Cx2Hx2W
    this is quite the same as the mask rcnn mask head in detectron2

    n_layers of 3x3 conv + relu -> 2x2 upsample (optional) -> 1x1 conv out_dim
    '''
    modules = []
    for i in range(n_layers):
        modules.append(
            Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.ReLU())
        )

    if upsample:
        modules.extend([
            ConvTranspose2d(in_dim, in_dim, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
        ])

    if out_dim is None:
        out_dim = in_dim
    modules.append(
        Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1))
    )

    module_seq = nn.Sequential(*modules)

    # init weights
    for i in range(len(module_seq)):
        if i < n_layers:
            weight_init.c2_msra_fill(module_seq[i])

    if upsample:
        weight_init.c2_msra_fill(module_seq[n_layers])

    nn.init.normal_(module_seq[-1].weight, std=0.001)
    if module_seq[-1].bias is not None:
        nn.init.constant_(module_seq[-1].bias, 0)

    return module_seq

def prepare_gt(pred_mask_logits, instances: List[Instances], mask_type=None):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if mask_type == 'invisible':
            gt_amodal_masks_per_image = instances_per_image.get('gt_amodal_masks').crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            gt_visible_masks_per_image = instances_per_image.get('gt_visible_masks').crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            gt_masks_per_image = gt_amodal_masks_per_image ^ gt_visible_masks_per_image
        else:
            gt_masks_per_image = instances_per_image.get(mask_type).crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if not cls_agnostic_mask:
        gt_classes = cat(gt_classes, dim=0)

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    return gt_masks, gt_masks_bool, gt_classes

def mask_loss(pred_mask_logits: torch.Tensor, 
              instances: List[Instances], vis_period: int = 0,
              mask_type=None):
    """
    Inherit mask loss from Mask R-CNN with mask_type param for amodal segmentation gt
    https://github.com/facebookresearch/detectron2/blob/0df924ce6066fb97d5413244614b12fbabaf65c8/detectron2/modeling/roi_heads/mask_head.py#L33
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    gt_masks, gt_masks_bool, gt_classes = prepare_gt(pred_mask_logits, instances, mask_type)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("shapeformer/accuracy_{}".format(mask_type), mask_accuracy)
    storage.put_scalar("shapeformer/false_positive_{}".format(mask_type), false_positive)
    storage.put_scalar("shapeformer/false_negative_{}".format(mask_type), false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss

def aware_loss(pred_mask_logits: torch.Tensor, 
              instances: List[Instances], vis_period: int = 0,
              mask_type=None):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    gt_masks, gt_masks_bool, gt_classes = prepare_gt(pred_mask_logits, instances, mask_type)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    bs = pred_mask_logits.size(0)
    pred_mask_logits = pred_mask_logits.view(bs, -1)
    log_input = F.log_softmax(pred_mask_logits, dim=1)
    log_target = F.softmax(gt_masks.view(bs, -1), dim=1)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    return 1 / kl_loss(log_input, log_target)

def mask_inference(pred_mask_logits: torch.Tensor, 
                   pred_instances: List[Instances],
                   pred_mask_type=None):
    """
    Inherit from detectron2 mask_rcnn_inference with prediction mask type
    https://github.com/facebookresearch/detectron2/blob/0df924ce6066fb97d5413244614b12fbabaf65c8/detectron2/modeling/roi_heads/mask_head.py#L33
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        device = (
            class_pred.device
            if torch.jit.is_scripting()
            else ("cpu" if torch.jit.is_tracing() else class_pred.device)
        )
        @torch.jit.script_if_tracing
        def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            """
            Tracing friendly way to cast tensor to another tensor's device. Device will be treated
            as constant during tracing, scripting the casting process as whole can workaround this issue.
            """
            return src.to(dst.device)
        indices = move_device_like(torch.arange(num_masks, device=device), class_pred)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.set(pred_mask_type, prob)  # (1, Hmask, Wmask)

def cls_loss(pred_cls_logits: torch.Tensor, 
              instances: List[Instances], vis_period: int = 0):
    """
    """
    gt_classes = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
        gt_classes.append(gt_classes_per_image)

    gt_classes = cat(gt_classes, dim=0)
    loss = F.cross_entropy(pred_cls_logits, gt_classes, reduction="mean")
    return loss
