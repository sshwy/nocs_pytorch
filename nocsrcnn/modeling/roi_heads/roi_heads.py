
from math import e
from typing import Dict
import torch
import numpy as np
from detectron2.layers import ShapeSpec, cat
from detectron2.layers.roi_align import ROIAlign
from detectron2.layers.mask_ops import (paste_mask_in_image_old,paste_masks_in_image)
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from nocsrcnn.modeling.roi_heads.nocs_head import *
import cv2

def clip_for_proposal(instances, out_hw, scale, ratio):
    target_mask, target_coords, target_class_ids, target_domain_labels, proposals_boxes \
            = [], [], [], [], []
    for image in instances:
        '''
        image.gt_masks
        image.gt_coords
        image.class_ids
        image.gt_domain_label
        image.proposals_boxes
        '''
        if len(image) == 0:
            continue

        target_class_ids.append(image.gt_classes)
        target_domain_labels.append(image.gt_domain_label)
        proposals_boxes.append(image.proposal_boxes)

        mask_per_image = image.gt_masks.crop_and_resize(
            image.proposal_boxes.tensor, out_hw
        )
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        target_mask.append(mask_per_image)

        coord_per_image = image.gt_coord
        bbox = image.proposal_boxes.tensor
        batch_inds = torch.arange(len(bbox)).to(dtype=bbox.dtype)[:, None]
        rois = torch.cat([batch_inds.cuda(), bbox], dim=1)  # Nx5



        coord_per_image = coord_per_image.permute(0,3,1,2)
       
        coord_per_image = ROIAlign((out_hw, out_hw), 1.0, 0, aligned=True).forward(coord_per_image, rois)
        coord_per_image = coord_per_image.permute(0,2,3,1)
        '''
        c = torch.zeros(coord_per_image[0].shape)
        c .copy_(coord_per_image[0])
        c=c*255
        b=c.cpu().numpy()
        cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/coord0.jpg',b)
        exit(0)
        '''
        target_coords.append(coord_per_image)

    return target_mask, target_coords, target_class_ids, target_domain_labels, proposals_boxes    
@ROI_HEADS_REGISTRY.register()
class NOCSRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for NOCS R-CNN
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self._init_nocs_head(cfg, input_shape)
        # self._vis = cfg.MODEL.VIS_MINIBATCH
        # self._misc = {}
        # self._vis_dir = cfg.OUTPUT_DIR

    def _init_nocs_head(self, cfg, input_shape):
        # fmt: off
        self.nocs_on        = cfg.MODEL.NOCS_ON
        if not self.nocs_on:
            return
        nocs_pooler_resolution = cfg.MODEL.ROI_NOCS_HEAD.POOLER_RESOLUTION#池化区域的输出大小，cfg设置，应该是14*14
        self.nocs_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)#池化操作相对于输入图像的比例
        self.nocs_sampling_ratio = cfg.MODEL.ROI_NOCS_HEAD.POOLER_SAMPLING_RATIO#ROIAlign运算参数
        nocs_pooler_type = cfg.MODEL.ROI_NOCS_HEAD.POOLER_TYPE#池化操作类型名称
        self.out_hw = cfg.MODEL.ROI_NOCS_HEAD.OUT_HW#输出的大小，通常是28
        self.num_bins = cfg.MODEL.ROI_NOCS_HEAD.NUM_BINS
        self.cfg = cfg

        self.COORD_USE_BINS=cfg.MODEL.ROI_NOCS_HEAD.COORD_USE_BINS
        self.COORD_SHARE_WEIGHTS=cfg.MODEL.ROI_NOCS_HEAD.COORD_SHARE_WEIGHTS
        self.USE_SYMMETRY_LOSS = cfg.MODEL.ROI_NOCS_HEAD.USE_SYMMETRY_LOSS
        self.COORD_USE_DELTA=cfg.MODEL.ROI_NOCS_HEAD.COORD_USE_DELTA
        self.COORD_REGRESS_LOSS = cfg.MODEL.ROI_NOCS_HEAD.COORD_REGRESS_LOSS
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]
        # print("******", nocs_pooler_resolution,nocs_pooler_scales,nocs_sampling_ratio,nocs_pooler_type,"*****")
        self.nocs_pooler = ROIPooler(
            output_size=nocs_pooler_resolution,
            scales=self.nocs_pooler_scales,
            sampling_ratio=self.nocs_sampling_ratio,
            pooler_type=nocs_pooler_type,
        )
        #print(nocs_pooler_resolution,self.nocs_pooler_scales,self.nocs_sampling_ratio,nocs_pooler_type)

            
        if self.COORD_SHARE_WEIGHTS :
            self.nocs_head_xyz = build_nocs_head(
                cfg,
                ShapeSpec(channels=in_channels, height=nocs_pooler_resolution, width=nocs_pooler_resolution),
                "coord_xyz",
            )
        else :
            self.nocs_head_x = build_nocs_head(
                cfg,
                ShapeSpec(channels=in_channels, height=nocs_pooler_resolution, width=nocs_pooler_resolution),
                "coord_x",
            )
            self.nocs_head_y = build_nocs_head(
                cfg,
                ShapeSpec(channels=in_channels, height=nocs_pooler_resolution, width=nocs_pooler_resolution),
                "coord_y",
            )
            self.nocs_head_z = build_nocs_head(
                cfg,
                ShapeSpec(channels=in_channels, height=nocs_pooler_resolution, width=nocs_pooler_resolution),
                "coord_z",
            )
    
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        # if self._vis:
        #    self._misc["images"] = images
        #print((targets[0].gt_coord.cpu().numpy().shape))
        
        del images
        if self.training:

            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_nocs(features, proposals))
            #losses = self._forward_nocs(features, proposals)
            return [], losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
    

    def forward_with_given_boxes(self, features, instances):
        assert not self.training
        
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes") 
        instances = self._forward_mask(features, instances)
        
        instances = self._forward_nocs(features, instances)
        #print(instances[0].pred_coord)
        '''
        for images in instances :
            print('1111')
            for i in range(images.pred_coord.shape[0]) :
                a = torch.zeros(images.pred_coord[i].shape)
                a .copy_(images.pred_coord[i])
                a = a*255
                a = a.int()
                b=a.cpu().numpy()
                cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/coord{}.jpg'.format(i),b)
            break 
        '''
        
        
        return instances

    def argmax_and_reshape(self,mrcnn_coord_bin):
        ## convert bins to float values
        mrcnn_coord_shape = mrcnn_coord_bin.shape#([128, 28, 28, 6, 32])
        mrcnn_coord_bin_reshape = mrcnn_coord_bin.contiguous().view(-1, mrcnn_coord_shape[-1]) #[702464, 32]
        mrcnn_coord_bin_value = torch.argmax(mrcnn_coord_bin_reshape, axis=-1).float()/float(self.num_bins)
        mrcnn_coord_bin_value = mrcnn_coord_bin_value.view(mrcnn_coord_shape[:-1])#[128, 28, 28, 6]
        mrcnn_coord_bin_value = mrcnn_coord_bin_value.permute(0, 3, 1, 2)
        return mrcnn_coord_bin_value
    def _forward_nocs(self, features, instances):
        '''

        '''
        if not self.nocs_on: 
            return {} if self.training else instances
        features = [features[f] for f in self.in_features]
        # assert instances[0].has("pred_nocs")
        if self.training:

            target_mask, target_coords, target_class_ids, target_domain_labels, proposals_boxes\
             = clip_for_proposal(instances, self.out_hw, self.nocs_pooler_scales, self.nocs_sampling_ratio)
            
            target_class_ids = torch.cat(target_class_ids)  #cat后为[128]

            target_mask = torch.cat(target_mask)
            target_coords = torch.cat(target_coords)
            target_coords_x ,target_coords_y,target_coords_z= target_coords[:,:,:,0],target_coords[:,:,:,1],target_coords[:,:,:,2]
            target_domain_labels = torch.cat(target_domain_labels) #  [128]
            all_true = torch.ones(target_domain_labels.shape)
            all_true = all_true.cuda()
            domain_ix = all_true-target_domain_labels
            if domain_ix[0] == 0 :
                target_class_ids = torch.ones(target_class_ids.shape)*(self.cfg.MODEL.ROI_NOCS_HEAD.NUM_CLASSES)
            #target_class_ids = torch.mul(target_class_ids, domain_ix.float())
            #features :         4 * batch_size * 256 * feature_map_size *f eature_map_size
            #proposals_boxes :  batch_size * batch_per_image(64) * 4
            nocs_features = self.nocs_pooler(features, proposals_boxes)
            if self.COORD_USE_BINS : 
                mrcnn_coord_x_bin, mrcnn_coord_x_feature = self.nocs_head_x(nocs_features)# mrcnn_coord_x_bin   ([128, 28, 28, 7, 32])
                mrcnn_coord_y_bin, mrcnn_coord_y_feature = self.nocs_head_y(nocs_features)
                mrcnn_coord_z_bin, mrcnn_coord_z_feature = self.nocs_head_z(nocs_features)
                mrcnn_coords_bin = torch.stack((mrcnn_coord_x_bin, mrcnn_coord_y_bin, mrcnn_coord_z_bin),dim=-1)
                if self.USE_SYMMETRY_LOSS :
                    coord_loss = mrcnn_coord_bins_symmetry_loss_graph( target_mask, target_coords, target_class_ids,target_domain_labels, mrcnn_coords_bin)
                    coord_x_loss ,coord_y_loss,coord_z_loss = coord_loss
                else :
                    1
                mrcnn_coord_x_bin_value = self.argmax_and_reshape(mrcnn_coord_x_bin)
                mrcnn_coord_y_bin_value = self.argmax_and_reshape(mrcnn_coord_y_bin)
                mrcnn_coord_z_bin_value = self.argmax_and_reshape(mrcnn_coord_z_bin)
                mrcnn_coords = torch.stack((mrcnn_coord_x_bin_value, mrcnn_coord_y_bin_value, mrcnn_coord_z_bin_value), dim=-1)
            else :
                if self.COORD_SHARE_WEIGHTS :
                    mrcnn_coords, _ = self.nocs_head_xyz(nocs_features)
                    mrcnn_coord_x = mrcnn_coords[:,:,:,:,0]
                    mrcnn_coord_y = mrcnn_coords[:,:,:,:,1]
                    mrcnn_coord_z = mrcnn_coords[:,:,:,:,2]
                else :

                    mrcnn_coord_x, mrcnn_coord_x_feature = self.nocs_head_x(nocs_features)
                    mrcnn_coord_y, mrcnn_coord_y_feature = self.nocs_head_y(nocs_features)
                    mrcnn_coord_z, mrcnn_coord_z_feature = self.nocs_head_z(nocs_features)
                    mrcnn_coords = torch.stack((mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z),dim=-1)
                if self.USE_SYMMETRY_LOSS :
                    '''
                    if self.COORD_REGRESS_LOSS == 'Soft_L1':
                        fn = mrcnn_coords_symmetry_smoothl1_loss_graph_3
                    elif self.COORD_REGRESS_LOSS == 'L1':
                        fn = mrcnn_coords_symmetry_l1_loss_graph_3
                        
                    elif self.COORD_REGRESS_LOSS == 'L2':
                        fn = mrcnn_coords_symmetry_l2_loss_graph_3
                    else:
                        assert False, 'wrong regression loss name!'
                    '''
                    if self.COORD_REGRESS_LOSS == 'Soft_L1':
                        fn = smoothl1_diff
                    elif self.COORD_REGRESS_LOSS == 'L1':
                        fn = l1_loss  
                    elif self.COORD_REGRESS_LOSS == 'L2':
                        fn = l2_loss
                    else:
                        assert False, 'wrong regression loss name!'
            
                    coord_loss = mrcnn_coords_symmetry_loss_graph_3( target_mask, target_coords, target_class_ids,target_domain_labels, mrcnn_coords,fn)
                    
                    coord_x_loss ,coord_y_loss,coord_z_loss = coord_loss
                else :
                    if self.COORD_REGRESS_LOSS == 'Soft_L1':
                        fn = mrcnn_coord_smoothl1_loss_graph_1
                    elif self.COORD_REGRESS_LOSS == 'L1':
                        fn =mrcnn_coord_l1_loss_graph_1
                        
                    elif self.COORD_REGRESS_LOSS == 'L2':
                        fn = mrcnn_coord_l2_loss_graph_1
                    else:
                        assert False, 'wrong regression loss name!'
                    coord_x_loss = fn(
                        target_mask, target_coords_x, target_class_ids,target_domain_labels, mrcnn_coord_x
                    )
                    coord_y_loss = fn(
                        target_mask, target_coords_y, target_class_ids,target_domain_labels, mrcnn_coord_y
                    )
                    coord_z_loss = fn(
                        target_mask, target_coords_z, target_class_ids,target_domain_labels, mrcnn_coord_z
                    )
                   
                    
            # mrcnn_coords_bin: [128, 28, 28, 7, 32, 3]
            
            #coord_bin_loss = mrcnn_coord_bins_symmetry_loss_graph(
            #    target_mask, target_coords, target_class_ids,target_domain_labels, mrcnn_coords_bin
            #)
            if self.USE_SYMMETRY_LOSS :
               # mrcnn_coords : [192, 6, 28, 28, 3]
                coord_l2_diff =mrcnn_coords_symmetry_l2_loss_graph_1(
                    target_mask, target_coords, target_class_ids,target_domain_labels, mrcnn_coords
                )
                coord_diff =mrcnn_coords_symmetry_l1_loss_graph_3(
                    target_mask, target_coords, target_class_ids,target_domain_labels, mrcnn_coords
                )
                coord_x_diff,coord_y_diff,coord_z_diff = coord_diff
            else :
                coord_l2_diff = mrcnn_coords_l2_loss_graph_1(
                    target_mask, target_coords, target_class_ids,target_domain_labels, mrcnn_coords
                )
                coord_x_diff =  mrcnn_coord_l1_loss_graph_1(
                    target_mask, target_coords_x, target_class_ids,target_domain_labels, mrcnn_coord_x
                )
                coord_y_diff =  mrcnn_coord_l1_loss_graph_1(
                    target_mask, target_coords_y, target_class_ids,target_domain_labels, mrcnn_coord_y
                )
                coord_z_diff =  mrcnn_coord_l1_loss_graph_1(
                    target_mask, target_coords_z, target_class_ids,target_domain_labels, mrcnn_coord_z
                )
            
            losses = {  "coord_x_loss":coord_x_loss,
                        "coord_y_loss":coord_y_loss,
                        "coord_z_loss":coord_z_loss,
                        "coord_l2_diff": coord_l2_diff,
                        "coord_x_diff":coord_x_diff,
                        "coord_y_diff":coord_y_diff ,
                        "coord_z_diff":coord_z_diff 
                        }
            return losses

        else:
           # print(instances)
            for image in instances  :
                boxes ,class_ids= [],[]
                class_ids=image.pred_classes
                inst_num =len(class_ids)
                boxes.append(image.pred_boxes)
                nocs_features = self.nocs_pooler(features, boxes) 
                if self.COORD_USE_BINS : 
                    mrcnn_coord_x_bin, mrcnn_coord_x_feature = self.nocs_head_x(nocs_features)
                    mrcnn_coord_y_bin, mrcnn_coord_y_feature = self.nocs_head_y(nocs_features)
                    mrcnn_coord_z_bin, mrcnn_coord_z_feature = self.nocs_head_z(nocs_features)
                    mrcnn_coords_bin = torch.stack((mrcnn_coord_x_bin, mrcnn_coord_y_bin, mrcnn_coord_z_bin),dim=-1)

                    mrcnn_coords_bin = mrcnn_coords_bin.permute(0,3,1,2,4,5)
                    #print(mrcnn_coords_bin.shape)#([2, 28, 28, 32, 3])
                    index = [i for i in range(inst_num)]
                    mrcnn_coords = mrcnn_coords_bin[index,class_ids,:,:,:,:]
                    #print(mrcnn_coords.shape)#([2, 28, 28, 32, 3])
                    mrcnn_coords = mrcnn_coords.argmax(dim=-2)
                    #print(mrcnn_coords.shape)#([2, 28, 28, 32, 3])
                    image.pred_coord =  mrcnn_coords.float()/float(self.num_bins)
                else :
                    if self.COORD_SHARE_WEIGHTS :
                        mrcnn_coords, _ = self.nocs_head_xyz(nocs_features)
                        #mrcnn_coord_x = mrcnn_coords[:,:,:,:,0]
                        #mrcnn_coord_y = mrcnn_coords[:,:,:,:,1]
                        #mrcnn_coord_z = mrcnn_coords[:,:,:,:,2]
                    else :

                        mrcnn_coord_x, mrcnn_coord_x_feature = self.nocs_head_x(nocs_features)
                        mrcnn_coord_y, mrcnn_coord_y_feature = self.nocs_head_y(nocs_features)
                        mrcnn_coord_z, mrcnn_coord_z_feature = self.nocs_head_z(nocs_features)
                        # config.USE_SYMMETRY_LOSS:
                        mrcnn_coords = torch.stack((mrcnn_coord_x, mrcnn_coord_y, mrcnn_coord_z),dim=-1)
                    #print(mrcnn_coords_bin.shape)#([2, 6, 28, 28, 3])
                    #print(class_ids.shape)
                    index = [i for i in range(inst_num)]
                    mrcnn_coords = mrcnn_coords[index,class_ids,:,:,:]
                    image.pred_coord =  mrcnn_coords
                '''
                mrcnn_coords = mrcnn_coords_bin[index,class_ids,:,:,:].cpu().numpy()
                im_coords = torch.zeros(inst_num,image.image_size[0],image.image_size[1],3)
                for i in range(inst_num) :
                    im_coord = unmold_coord(mrcnn_coords[i,:,:,:],list(image.pred_boxes[i].tensor.int().cpu().numpy()[0]),image.image_size)
                    
                    a = im_coord
                    a=a*255
                    d = a.astype(int)
                    cv2.imwrite('/home/qiweili/detectron2/projects/dect-demo/mask/pred_coord{}.jpg'.format(i),d)
                    
                    im_coords[i,:,:,:] = torch.from_numpy(im_coord)
                image.pred_coord = im_coords
                '''    
                #exit(0)
                print(1)
            #exit(0)
            return instances