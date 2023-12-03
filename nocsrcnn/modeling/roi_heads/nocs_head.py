# import fvcore.nn.weight_init as weight_init
import cv2
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F
import sys
import math
import numpy as np
import fvcore.nn.weight_init as weight_init

# from meshrcnn.structures.voxel import batch_crop_voxels_within_box

ROI_NOCS_HEAD_REGISTRY = Registry("ROI_NOCS_HEAD")


def clip_by_tensor(t, t_min, t_max):
    # t  [128, 28, 28, 6, 3]

    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max

    return result


def class_id_to_theta(class_id, *y):
    #     synset_names = ['bottle', #0,'bowl', #1,'camera', #2
    #                 'can',  #3,'laptop',#4.'mug'#5
    #                 ]
    if class_id in [0, 1, 3]:
        return 2 * math.pi / 6
    else:
        return 0.0


def mrcnn_coord_symmetry_loss_graph(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):
    """Mask L1 loss for the coordinates head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: [batch, num_rois, height, width, 3].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_domain_labels: [batch, num_rois]. Bool. 1 for real data, 0 for synthetic data.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, 3] float32 tensor with values from 0 to 1.
    """
    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    pred_coords = pred_coords.permute(0, 3, 1, 2, 4)

    pos_ix = torch.where(target_class_ids < 6)[0]

    if pos_ix.shape[0] <= 0:
        return torch.zeros(1, 3)

    pos_class_ids = torch.gather(target_class_ids, 0, pos_ix).int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()
    temp = torch.zeros(pos_class_ids.shape)
    temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    pos_class_ids.copy_(temp)
    pos_class_ids = pos_class_ids.cuda()
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]

    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)
    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    y_true_stack = y_true_stack + 0.5

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]
    y_pred = y_pred.unsqueeze(3)
    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)

    diff = torch.abs(
        y_true_stack - y_pred_stack
    )  ## shape: [num_pos_rois, height, width, 6, 3]

    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)
    ## shape: [num_pos_rois, height, width, 1, 1]

    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]

    diff_in_mask = diff.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = diff_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]

    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]

    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]

    num_of_pixels = num_of_pixels.unsqueeze(1)
    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]
    sym_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    return sym_loss


def mrcnn_coord_symmetry_euclidean_distance_graph(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):
    """Mask euclidean distance for the coordinates head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: [batch, num_rois, height, width, 3].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_coords: [batch, proposals, height, width, num_classes, 3] float32 tensor with values from 0 to 1.
    """
    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    pred_coords = pred_coords.permute(0, 3, 1, 2, 4)

    pos_ix = torch.where(target_class_ids < 6)[0]
    if pos_ix.shape[0] <= 0:
        return torch.zeros(1)

    pos_class_ids = torch.gather(target_class_ids, 0, pos_ix).int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()
    temp = torch.zeros(pos_class_ids.shape)
    temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    pos_class_ids.copy_(temp)
    pos_class_ids = pos_class_ids.cuda()
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]

    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)
    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    y_true_stack = y_true_stack + 0.5

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]
    y_pred = y_pred.unsqueeze(3)
    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)

    diff = torch.abs(
        y_true_stack - y_pred_stack
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    diff = torch.square(diff)  ## shape: [num_pos_rois, height, width, 6, 3]

    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)
    ## shape: [num_pos_rois, height, width, 1, 1]

    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]

    diff_in_mask = diff.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = diff_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]

    min_squared_diff_sum_in_mask, _ = total_sum_loss_in_mask.min(dim=-1)
    mean_squared_diff_sum_in_mask = min_squared_diff_sum_in_mask.div(
        num_of_pixels
    )  ## shape: [num_pos_rois]
    euclidean_dist_in_mask = mean_squared_diff_sum_in_mask.sqrt()

    dist = euclidean_dist_in_mask.mean(dim=0)  ## shape:[1]

    return dist


def mrcnn_coord_bins_symmetry_loss_graph_buyongle(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):
    """Mask L2 loss for the coordinates head.
    target_masks: list(batch)[BitMasks()]
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: list(batch)[Tensor(num_rois, height, width, 3)].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: list(batch)[Tensor(num_rois)].
        Integer class IDs. Zero padded.
    pred_coords: Tensor[batch*proposals, height, width, num_classes, num_bins, 3]
        float32 tensor with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    # 发现了这行注释之后发现直接合并就好了
    num_bins = pred_coords.shape[-2]
    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    pred_coords = pred_coords.permute(0, 3, 1, 2, 5, 4)  # [128, 7, 28, 28, 3, 32]

    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3)

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :, :
    ]  # pred_coords[indices]

    ## shape: [num_pos_rois, height, width,  3, num_bins]

    y = torch.ones(1).long().cuda()
    # y = y*0.5
    y = y.repeat(y_pred.shape[0], 28, 28, 3)
    # print(y.shape)
    y_pred_stack = y_pred.permute(0, 4, 1, 2, 3)
    # loss = y-y_pred_stack
    # sum_loss_in_mask = loss.sum(dim=[1,2,3])
    # with torch.no_grad():
    #    sum_loss_in_mask = sum_loss_in_mask/(28*64)
    cross_loss = F.cross_entropy(y_pred_stack, y, reduction="none")
    ## shape: [num_pos_rois, height, width,  3]

    sum_loss_in_mask = cross_loss.sum(dim=[1, 2])  ## shape: [num_pos_rois,  3]
    sym_loss = sum_loss_in_mask.mean(dim=0)  ## shape:[3]
    # sym_loss=sym_loss.div(600)
    # with torch.no_grad():
    #    sym_loss=sym_loss.mul(2)

    # exit(0)
    sym_loss = sym_loss / (28 * 28)
    return sym_loss


def mrcnn_coord_bins_symmetry_loss_graph(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):
    """Mask L2 loss for the coordinates head.
    target_masks: list(batch)[BitMasks()]
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_coords: list(batch)[Tensor(num_rois, height, width, 3)].
        A float32 tensor of values in the range of [0, 1]. Uses zero padding to fill array.
    target_class_ids: list(batch)[Tensor(num_rois)].
        Integer class IDs. Zero padded.
    pred_coords: Tensor[batch*proposals, height, width, num_classes, num_bins, 3]
        float32 tensor with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    # 发现了这行注释之后发现直接合并就好了
    num_bins = pred_coords.shape[-2]
    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    pred_coords = pred_coords.permute(0, 3, 1, 2, 5, 4)  # [128, 7, 28, 28, 3, 32]

    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]

    if pos_ix.shape[0] <= 0:
        return torch.zeros(3).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :, :
    ]  # pred_coords[indices]

    # temp = torch.zeros(pos_class_ids.shape)
    # temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    # pos_class_ids.copy_(temp)
    # pos_class_ids = pos_class_ids.cuda()  #旋转的角度 [128] ,同pos_class_rotation_theta
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    # 旋转矩阵
    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]
    # pos_class_rotation_matrix[0] :  [[ 0.5403,  0.0000, -0.8415],[ 0.0000,  1.0000,  0.0000],[ 0.8415,  0.0000,  0.5403]]
    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    # 获得6个旋转角度后的true
    y_true_stack = y_true_stack + 0.5
    # 将true按bin扩展
    y_true_bins_stack = y_true_stack * (float(num_bins)) - 0.000001
    y_true_bins_stack = y_true_bins_stack.floor().int()

    y_true_bins_stack = clip_by_tensor(y_true_bins_stack, 0, num_bins - 1).long()

    # pred_coords [128, 7, 28, 28, 3, 32]

    y_pred = y_pred.unsqueeze(3)

    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1, 1)
    ## shape: [num_pos_rois, height, width, 6, 3, num_bins]

    y_pred_stack = y_pred_stack.permute(0, 5, 1, 2, 3, 4)

    cross_loss = F.cross_entropy(y_pred_stack, y_true_bins_stack, reduction="none")
    ## shape: [num_pos_rois, height, width, 6, 3]

    y_pred_stack = y_pred_stack.permute(0, 2, 3, 4, 5, 1)
    """
    print(pos_class_ids)
    print(pos_ix)
    for i in range(pos_class_ids.shape[0]) :
        a = torch.zeros(target_masks[pos_ix[i]].shape)
        a .copy_(target_masks[pos_ix[i]].int())
        a = a*255
        a = a.repeat(3,1,1).permute(1,2,0)
        b=a.cpu().numpy()
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/mask{}.jpg'.format(i),b)
    exit(0)
    """
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)
    ## shape: [num_pos_rois, height, width, 1, 1]

    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]
    # print(num_of_pixels)
    cross_loss_in_mask = cross_loss.mul(
        reshape_mask
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = cross_loss_in_mask.sum(
        dim=[1, 2]
    )  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]

    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]
    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    # print(arg_min_rotation)
    # exit(0)
    # print(sum_loss_in_mask.shape)
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]
    # print(min_loss_in_mask.shape)

    num_of_pixels = num_of_pixels.unsqueeze(1)
    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]
    sym_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    return sym_loss


def mrcnn_coords_reg_loss_graph(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3)

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1)

    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    # diff = torch.abs(y_pred_inmask,y_true_inmask )\
    """
    for i in range(pos_class_ids.shape[0]) :

        c = torch.zeros(y_true_inmask[i].shape)
        a = torch.zeros(y_true_inmask[i].shape)
        c.copy_(y_true_inmask[i])
        a.copy_(y_pred_inmask[i])
        c=c*255
        a=a*255
        b=c.detach().numpy()
        d=a.detach().numpy()
        #print(np.unique(b))
        #exit(0)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/coord{}.jpg'.format(i),b)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/pred_coord{}.jpg'.format(i),d)
        print(i)
    exit(0)
    """

    print(y_pred.max(), y_pred.min(), y_true.max(), y_true.min())
    loss = F.smooth_l1_loss(y_true_inmask, y_pred_inmask)
    # print(loss)
    # exit(0)
    return loss


def mrcnn_coord_reg_loss_graph(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3)

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2)  #  [128, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights)  #   [128, 28, 28]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :
    ]  # pred_coords[indices]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2])

    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    # diff = torch.abs(y_pred_inmask,y_true_inmask )\
    """
    for i in range(pos_class_ids.shape[0]) :

        c = torch.zeros(y_true_inmask[i].shape)
        a = torch.zeros(y_true_inmask[i].shape)
        c.copy_(y_true_inmask[i])
        a.copy_(y_pred_inmask[i])
        c=c*255
        a=a*255
        b=c.detach().numpy()
        d=a.detach().numpy()
        #print(np.unique(b))
        #exit(0)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/coord{}.jpg'.format(i),b)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/pred_coord{}.jpg'.format(i),d)
        print(i)
    exit(0)
    """

    print(y_pred.max(), y_pred.min(), y_true.max(), y_true.min())
    loss = F.smooth_l1_loss(y_true_inmask, y_pred_inmask)
    # print(loss)
    # exit(0)
    return loss


def mrcnn_coord_l1_loss_graph_1(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(1).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2)  #  [128, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights)  #   [128, 28, 28]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :
    ]  # pred_coords[indices]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2])

    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    # diff = torch.abs(y_pred_inmask,y_true_inmask )\

    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    loss = F.l1_loss(y_true_inmask, y_pred_inmask)
    # print(loss)
    # exit(0)
    return loss


def mrcnn_coord_l2_loss_graph_1(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(1).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2)  #  [128, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights)  #   [128, 28, 28]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :
    ]  # pred_coords[indices]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2])

    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    # diff = torch.abs(y_pred_inmask,y_true_inmask )\

    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    loss = F.mse_loss(y_true_inmask, y_pred_inmask)
    # print(loss)
    # exit(0)
    return loss


def mrcnn_coord_smoothl1_loss_graph_1(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(1).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2)  #  [128, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights)  #   [128, 28, 28]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :
    ]  # pred_coords[indices]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2])

    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    # diff = torch.abs(y_pred_inmask,y_true_inmask )\

    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    loss = F.smooth_l1_loss(y_true_inmask, y_pred_inmask, beta=0.1)
    # print(loss)
    # exit(0)
    return loss


def mrcnn_coords_l2_loss_graph_1(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(1).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1)

    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    """
    for i in range(pos_class_ids.shape[0]) :

        c = torch.zeros(y_true_inmask[i].shape)
        a = torch.zeros(y_true_inmask[i].shape)
        c.copy_(y_true_inmask[i])
        a.copy_(y_pred_inmask[i])
        print(a.max())
        c=c*255
        a=a*255
        b=c.detach().numpy()
        d=a.detach().numpy()
        #print(np.unique(b))
        #exit(0)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/coord{}.jpg'.format(i),b)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/pred_coord{}.jpg'.format(i),d)
        print(a.max())
    exit(0)
    """

    # diff = torch.abs(y_pred_inmask,y_true_inmask )\

    print(y_pred.max(), y_pred.min(), y_true.max(), y_true.min())
    loss = F.mse_loss(y_true_inmask, y_pred_inmask)
    # print(loss)
    # exit(0)
    return loss


def smoothl1_diff(y_true, y_pred, threshold=0.1):
    diff = torch.abs(torch.sub(input=y_true, alpha=1, other=y_pred))
    coefficient = 1 / (2 * threshold)
    less = torch.pow(diff, 2) * coefficient
    more = diff - threshold / 2
    loss = torch.where(diff < threshold, less, more)
    return loss


def l1_loss(y_true, y_pred):
    diff = torch.abs(torch.sub(input=y_true, alpha=1, other=y_pred))
    return diff


def l2_loss(y_true, y_pred):
    diff = torch.abs(torch.sub(input=y_true, alpha=1, other=y_pred))
    loss = torch.pow(diff, 2)
    return loss


def mrcnn_coords_symmetry_loss_graph_3(
    target_masks,
    target_coords,
    target_class_ids,
    target_domain_labels,
    pred_coords,
    loss_fn,
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]

    # temp = torch.zeros(pos_class_ids.shape)
    # temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    # pos_class_ids.copy_(temp)
    # pos_class_ids = pos_class_ids.cuda()  #旋转的角度 [128] ,同pos_class_rotation_theta
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    # 旋转矩阵
    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]
    # pos_class_rotation_matrix[0] :  [[ 0.5403,  0.0000, -0.8415],[ 0.0000,  1.0000,  0.0000],[ 0.8415,  0.0000,  0.5403]]
    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    # 获得6个旋转角度后的true
    y_true_stack = y_true_stack + 0.5

    y_pred = y_pred.unsqueeze(3)

    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)

    loss = loss_fn(y_true_stack, y_pred_stack)

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]
    # print(num_of_pixels )
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)

    loss_in_mask = loss.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]
    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]
    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    num_of_pixels = num_of_pixels.unsqueeze(1)

    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]
    final_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    return final_loss


def mrcnn_coords_symmetry_smoothl1_loss_graph_3(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]

    # temp = torch.zeros(pos_class_ids.shape)
    # temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    # pos_class_ids.copy_(temp)
    # pos_class_ids = pos_class_ids.cuda()  #旋转的角度 [128] ,同pos_class_rotation_theta
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    # 旋转矩阵
    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]
    # pos_class_rotation_matrix[0] :  [[ 0.5403,  0.0000, -0.8415],[ 0.0000,  1.0000,  0.0000],[ 0.8415,  0.0000,  0.5403]]
    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    # 获得6个旋转角度后的true
    y_true_stack = y_true_stack + 0.5

    y_pred = y_pred.unsqueeze(3)

    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)
    loss1 = torch.abs(
        torch.sub(input=y_true_stack, alpha=1, other=y_pred_stack)
    )  # [ roi ,28 ,28 ,6 ,3]
    loss2 = loss1 - 0.05
    loss3 = (
        torch.pow(
            torch.abs(torch.sub(input=y_true_stack, alpha=1, other=y_pred_stack)), 2
        )
        * 5
    )
    loss = torch.where(loss2 > 0.05, loss2, loss3)
    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]
    # print(num_of_pixels )
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)

    loss_in_mask = loss.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]
    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]
    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    num_of_pixels = num_of_pixels.unsqueeze(1)

    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]
    final_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    return final_loss


def mrcnn_coords_symmetry_l1_loss_graph_3(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]

    # temp = torch.zeros(pos_class_ids.shape)
    # temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    # pos_class_ids.copy_(temp)
    # pos_class_ids = pos_class_ids.cuda()  #旋转的角度 [128] ,同pos_class_rotation_theta
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    # 旋转矩阵
    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]
    # pos_class_rotation_matrix[0] :  [[ 0.5403,  0.0000, -0.8415],[ 0.0000,  1.0000,  0.0000],[ 0.8415,  0.0000,  0.5403]]
    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    # 获得6个旋转角度后的true
    y_true_stack = y_true_stack + 0.5

    y_pred = y_pred.unsqueeze(3)

    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)
    loss = torch.abs(
        torch.sub(input=y_true_stack, alpha=1, other=y_pred_stack)
    )  # [ roi ,28 ,28 ,6 ,3]
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]
    # print(num_of_pixels )
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)

    loss_in_mask = loss.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]
    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]
    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    num_of_pixels = num_of_pixels.unsqueeze(1)

    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]
    final_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    return final_loss


def mrcnn_coords_symmetry_l2_loss_graph_3(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(3).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]

    # temp = torch.zeros(pos_class_ids.shape)
    # temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    # pos_class_ids.copy_(temp)
    # pos_class_ids = pos_class_ids.cuda()  #旋转的角度 [128] ,同pos_class_rotation_theta
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    # 旋转矩阵
    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]
    # pos_class_rotation_matrix[0] :  [[ 0.5403,  0.0000, -0.8415],[ 0.0000,  1.0000,  0.0000],[ 0.8415,  0.0000,  0.5403]]
    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    # 获得6个旋转角度后的true
    y_true_stack = y_true_stack + 0.5

    y_pred = y_pred.unsqueeze(3)

    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)
    loss = torch.pow(
        torch.abs(torch.sub(input=y_true_stack, alpha=1, other=y_pred_stack)), 2
    )  # [ roi ,28 ,28 ,6 ,3]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]
    # print(num_of_pixels )
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)

    loss_in_mask = loss.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]
    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]
    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    num_of_pixels = num_of_pixels.unsqueeze(1)

    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]
    final_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    return final_loss


def mrcnn_coords_symmetry_l2_loss_graph_1(
    target_masks, target_coords, target_class_ids, target_domain_labels, pred_coords
):

    heights = target_masks.shape[1]
    weights = target_masks.shape[2]
    # print(pred_coords.shape)
    # pred_coords = pred_coords.permute(0, 2,3,4,1)  #[128, 6, 28, 28, 3]
    # print(pred_coords.shape)
    # pred为物体的roi的索引
    pos_ix = torch.where(target_class_ids < 6)[0]  # target_class_ids : [128]
    # print(target_class_ids)
    if pos_ix.shape[0] <= 0:
        return torch.zeros(1).cuda()

    pos_class_ids = torch.gather(
        target_class_ids, 0, pos_ix
    )  # .int()  # [num_pos_rois]
    pos_class_ids = pos_class_ids.cpu()

    y_pred = pred_coords[
        pos_ix.long(), pos_class_ids.long(), :, :, :
    ]  # pred_coords[indices]
    # temp = torch.zeros(pos_class_ids.shape)
    # temp.copy_(pos_class_ids)
    pos_class_rotation_theta = pos_class_ids.map_(
        pos_class_ids, class_id_to_theta
    )  # 旋转的角度 [128]
    pos_class_rotation_theta = pos_class_rotation_theta.cuda()
    # pos_class_ids.copy_(temp)
    # pos_class_ids = pos_class_ids.cuda()  #旋转的角度 [128] ,同pos_class_rotation_theta
    pos_class_cos = torch.cos(pos_class_rotation_theta)
    pos_class_sin = torch.sin(pos_class_rotation_theta)
    pos_class_one = torch.ones(pos_class_rotation_theta.shape).cuda()
    pos_class_zero = torch.zeros(pos_class_rotation_theta.shape).cuda()
    line_1 = torch.stack((pos_class_cos, pos_class_zero, pos_class_sin), dim=1)
    line_2 = torch.stack((pos_class_zero, pos_class_one, pos_class_zero), dim=1)
    line_3 = torch.stack((pos_class_sin.neg(), pos_class_zero, pos_class_cos), dim=1)
    pos_class_rotation_matrix = torch.stack((line_1, line_2, line_3), dim=2)

    # 旋转矩阵
    pos_class_rotation_matrix = pos_class_rotation_matrix.view(
        -1, 1, 1, 3, 3
    )  # [num_pos_rois, 1, 1, 3, 3]
    # pos_class_rotation_matrix[0] :  [[ 0.5403,  0.0000, -0.8415],[ 0.0000,  1.0000,  0.0000],[ 0.8415,  0.0000,  0.5403]]
    tiled_rotation_matrix = pos_class_rotation_matrix.repeat(1, heights, weights, 1, 1)
    # [num_pos_rois, height, weigths, 3, 3]

    # Gather the coordinate maps and masks (predicted and true) that contribute to loss
    # true coord map:[N', height, width]
    pos_ix_ = pos_ix.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #  [128, 1, 1, 1]
    pos_ix_ = pos_ix_.expand(pos_ix.shape[0], heights, weights, 3)  #   [128, 28, 28, 3]

    y_true = torch.gather(
        target_coords, 0, pos_ix_
    )  ## shape: [num_pos_rois, height, width, 3]

    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1)
    """
    y_pred_inmask = y_pred.mul(reshape_mask)
    y_true_inmask = y_true.mul(reshape_mask)
    for i in range(pos_class_ids.shape[0]) :

        c = torch.zeros(y_true[i].shape)
        a = torch.zeros(y_true[i].shape)
        c.copy_(y_true_inmask[i])
        a.copy_(y_pred_inmask[i])
        print(a.max())
        c=c*255
        a=a*255
        b=c.detach().numpy()
        d=a.detach().numpy()
        #print(np.unique(b))
        #exit(0)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/coord{}.jpg'.format(i),b)
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/pred_coord{}.jpg'.format(i),d)
        print(a.max())
    #exit(0)
    """

    print(
        "x: ",
        round(y_pred[:, :, :, 0].max().item(), 4),
        round(y_pred[:, :, :, 0].min().item(), 4),
        round(y_true[:, :, :, 0].max().item(), 4),
        round(y_true[:, :, :, 0].min().item(), 4),
        "  y: ",
        round(y_pred[:, :, :, 1].max().item(), 4),
        round(y_pred[:, :, :, 1].min().item(), 4),
        round(y_true[:, :, :, 1].max().item(), 4),
        round(y_true[:, :, :, 1].min().item(), 4),
        "  z: ",
        round(y_pred[:, :, :, 2].max().item(), 4),
        round(y_pred[:, :, :, 2].min().item(), 4),
        round(y_true[:, :, :, 2].max().item(), 4),
        round(y_true[:, :, :, 2].min().item(), 4),
    )
    y_true = y_true - 0.5
    y_true = y_true.unsqueeze(4)  ## shape: [num_pos_rois, height, width, 3, 1]

    rotated_y_true_1 = torch.matmul(tiled_rotation_matrix, y_true)
    rotated_y_true_2 = torch.matmul(tiled_rotation_matrix, rotated_y_true_1)
    rotated_y_true_3 = torch.matmul(tiled_rotation_matrix, rotated_y_true_2)
    rotated_y_true_4 = torch.matmul(tiled_rotation_matrix, rotated_y_true_3)
    rotated_y_true_5 = torch.matmul(tiled_rotation_matrix, rotated_y_true_4)
    y_true_stack = torch.cat(
        (
            y_true,
            rotated_y_true_1,
            rotated_y_true_2,
            rotated_y_true_3,
            rotated_y_true_4,
            rotated_y_true_5,
        ),
        dim=4,
    )  ## shape: [num_pos_rois, height, width, 3, 6]
    y_true_stack = y_true_stack.permute(
        0, 1, 2, 4, 3
    )  ## shape: [num_pos_rois, height, width, 6, 3]
    # 获得6个旋转角度后的true
    y_true_stack = y_true_stack + 0.5

    y_pred = y_pred.unsqueeze(3)

    y_pred_stack = y_pred.repeat(1, 1, 1, y_true_stack.shape[3], 1)

    loss = torch.pow(
        torch.abs(torch.sub(input=y_true_stack, alpha=1, other=y_pred_stack)), 2
    )  # [ roi ,28 ,28 ,6 ,3]

    # exit(0)
    mask = target_masks[pos_ix]  ## shape: [num_pos_rois, height, width]
    num_of_pixels = mask.sum(dim=[1, 2]) + 0.00001  ## shape: [num_pos_rois]
    reshape_mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1)

    loss_in_mask = loss.mul(reshape_mask)  ## shape: [num_pos_rois, height, width, 6, 3]
    sum_loss_in_mask = loss_in_mask.sum(dim=[1, 2])  ## shape: [num_pos_rois, 6, 3]
    total_sum_loss_in_mask = sum_loss_in_mask.sum(dim=-1)  ## shape: [num_pos_rois, 6]
    arg_min_rotation = torch.argmin(
        total_sum_loss_in_mask, dim=-1
    ).long()  ##shape: [num_pos_rois]
    index_1 = torch.arange(0, arg_min_rotation.shape[0]).cuda().long()
    min_loss_in_mask = sum_loss_in_mask[
        index_1, arg_min_rotation, :
    ]  ## shape: [num_pos_rois, 3]
    # print(y_pred .max(),y_pred .min(),y_true .max(),y_true .min())
    num_of_pixels = num_of_pixels.unsqueeze(1)

    mean_loss_in_mask = min_loss_in_mask.div(
        num_of_pixels.expand_as(min_loss_in_mask)
    )  ## shape: [num_pos_rois, 3]

    final_loss = mean_loss_in_mask.mean(dim=0)  ## shape:[3]
    euclidean_dist_in_mask = torch.sqrt(final_loss)
    final_loss = euclidean_dist_in_mask.mean(dim=0)

    return final_loss


@ROI_NOCS_HEAD_REGISTRY.register()
class NOCSRCNNGraphConvSubdHead(nn.Module):
    """
    8个卷积层14*14*256->14*14*256
    上采样->28*28*256
    ->28*28*(B)N
    源代码里面build_fpn_coord_bins_graph这个函数，出去pooling后半段
    """

    def __init__(self, cfg, input_shape, net_name):
        super(NOCSRCNNGraphConvSubdHead, self).__init__()
        # fmt:off
        num_conv           = cfg.MODEL.ROI_NOCS_HEAD.NUM_CONV # 刚开始有几个卷积层 论文中4个
        self.num_bins      = cfg.MODEL.ROI_NOCS_HEAD.NUM_BINS #这是要多少个bins
        self.num_classes        = cfg.MODEL.ROI_NOCS_HEAD.NUM_CLASSES
        input_channels     = input_shape.channels
        self.norm          = cfg.MODEL.ROI_NOCS_HEAD.NORM
        self.USE_SYMMETRY_LOSS      = cfg.MODEL.ROI_NOCS_HEAD.USE_SYMMETRY_LOSS
        self.COORD_USE_BINS         = cfg.MODEL.ROI_NOCS_HEAD.COORD_USE_BINS
        self.COORD_SHARE_WEIGHTS    = cfg.MODEL.ROI_NOCS_HEAD.COORD_SHARE_WEIGHTS
        self.COORD_USE_DELTA        = cfg.MODEL.ROI_NOCS_HEAD.COORD_USE_DELTA
        self.USE_BN                 = cfg.MODEL.ROI_NOCS_HEAD.USE_BN     
        # fmt:on

        self.conv_norm_relus = []  # 前面4个卷积层
        self.batch_norm = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels,  # input-channels在这里一直保持256
                input_channels,  # output-channels在这里也一直保持256
                kernel_size=3,
                stride=1,
                padding=1,
                # bias=not self.norm,
                # norm=get_norm(self.norm, input_channels),
                # activation=F.relu,
            )
            self.add_module("mrcnn_{}_conv".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            bn = nn.BatchNorm2d(input_channels, affine=False)
            self.add_module("mrcnn_{}_bn".format(k + 1), bn)
            self.batch_norm.append(bn)

        self.deconv = ConvTranspose2d(
            input_channels,  # input-channels在这里一直保持256
            input_channels,  # output-channels在这里也一直保持256
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.add_module("mrcnn_deconv", self.deconv)

        # print(self.num_bins,num_classes)
        self.conv_reg = Conv2d(
            input_channels,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.add_module("mrcnn_conv_reg", self.conv_reg)

        self.conv_reg3 = Conv2d(
            input_channels,
            self.num_classes * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.add_module("mrcnn_conv_reg3", self.conv_reg3)

        self.conv_bins = Conv2d(
            input_channels,
            self.num_classes * self.num_bins,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.add_module("mrcnn_conv_bins", self.conv_bins)
        for layer in self.conv_norm_relus + [self.deconv] + [self.conv_bins]:
            weight_init.c2_xavier_fill(layer)

    def build_bin_coord_graph(self, x):
        for layer, bn in zip(self.conv_norm_relus, self.batch_norm):
            # print("*****",x.shape,"======")
            x = layer(x)
            if self.USE_BN:
                x = bn(x)
            x = F.relu(x)

        x = self.deconv(x)

        x_feature = F.relu(
            x
        )  # (batch_size*batch_per_image(64)) * 256 * 28 * 28     [128, 256, 28, 28]
        x = self.conv_bins(
            x_feature
        )  # (batch_size*batch_per_image(64)) * (class_num*bin_num) * 28 * 28   [128, 224, 28, 28]

        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), x.size(1), x.size(2), self.num_classes, self.num_bins)

        # x = F.softmax(x,dim=-1)
        return x, x_feature

    def build_regress_coord_graph(self, x):
        for layer, bn in zip(self.conv_norm_relus, self.batch_norm):
            # print("*****",x.shape,"======")
            x = layer(x)
            if self.USE_BN:
                x = bn(x)
            x = F.relu(x)

        x = self.deconv(x)

        x_feature = F.relu(
            x
        )  # (batch_size*batch_per_image(64)) * 256 * 28 * 28     [128, 256, 28, 28]
        x = self.conv_reg(
            x_feature
        )  # (batch_size*batch_per_image(64)) * class_num * 28 * 28   [128, 224, 28, 28]
        # print(self.conv_bins.weight[0].max())
        return x, x_feature

    def build_regress_coords_graph(self, x):
        for layer, bn in zip(self.conv_norm_relus, self.batch_norm):
            # print("*****",x.shape,"======")
            x = layer(x)
            if self.USE_BN:
                x = bn(x)
            x = F.relu(x)

        x = self.deconv(x)

        x_feature = F.relu(
            x
        )  # (batch_size*batch_per_image(64)) * 256 * 28 * 28     [128, 256, 28, 28]
        x = self.conv_reg3(
            x_feature
        )  # (batch_size*batch_per_image(64)) * (class_num*3) * 28 * 28   [128, 224, 28, 28]
        # print(self.conv_bins.weight[0].max())
        x = x.reshape(x.size(0), -1, x.size(2), x.size(3), 3)
        return x, x_feature

    def forward(self, x):
        # print("*****",x.shape,"======")
        if self.COORD_USE_BINS:
            x, x_feature = self.build_bin_coord_graph(x)
        else:
            if self.COORD_SHARE_WEIGHTS:  # 初始化一个head
                x, x_feature = self.build_regress_coords_graph(x)
            else:  # 初始化三个head
                x, x_feature = self.build_regress_coord_graph(x)

        return x, x_feature


def build_nocs_head(cfg, input_shape, net_name):
    name = cfg.MODEL.ROI_NOCS_HEAD.NAME
    return ROI_NOCS_HEAD_REGISTRY.get(name)(cfg, input_shape, net_name)
