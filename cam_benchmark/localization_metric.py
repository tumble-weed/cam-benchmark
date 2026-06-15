import torch
'''
# read the bbox for a image


# get the pred-bbox
# calculate iou
# if iou > hit_thresh, mark as hit

def iou( 
pred_bbox_tlbr_x0y0x1y1, #[x_tl,y_tl, x_br,y_br]
gt_bbox_tlbr_x0y0x1y1, 
):
    ...
    # find some library that has a tested implementation
    return iou

def localization_metric(
pred_bbox_tlbr_x0y0x1y1, #[x_tl,y_tl, x_br,y_br]
gt_bbox_tlbr_x0y0x1y1, 
hit_thresh_iou,
):
    iou_ = iou(...)
    hit = ...
    return {
    'hit':...,
    'iou':iou_,
    }

'''


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps=1e-7):
    """
    pred_mask, gt_mask: binary tensors (0/1) of same shape
    """
    intersection = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum()
    return intersection / (union + eps)

