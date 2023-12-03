from detectron2.config import CfgNode as CN

def get_nocsrcnn_cfg_defaults(cfg):
    """
    Customize the detectron2 cfg to include some new keys and default values
    for NOCS branch
    """
    #接下来可以尽情往这里面丢有关这个NOCS分支的所有cfg

    cfg.MODEL.NOCS_ON = False
    cfg.MODEL.VIS_MINIBATCH = False  # visualize minibatches

    # aspect ratio grouping has no difference in performance
    # but might reduce memory by a little bit
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    # ------------------------------------------------------------------------ #
    # Mesh Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_NOCS_HEAD = CN()
    cfg.MODEL.ROI_NOCS_HEAD.NAME = "NOCSRCNNGraphConvSubdHead"

    cfg.MODEL.ROI_NOCS_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_NOCS_HEAD.OUT_HW = 28
    cfg.MODEL.ROI_NOCS_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_NOCS_HEAD.POOLER_TYPE = "ROIAlign"
    # Numer of stages
    cfg.MODEL.ROI_NOCS_HEAD.NUM_STAGES = 1
    cfg.MODEL.ROI_NOCS_HEAD.NUM_GRAPH_CONVS = 1  # per stage
    cfg.MODEL.ROI_NOCS_HEAD.GRAPH_CONV_DIM = 256
    cfg.MODEL.ROI_NOCS_HEAD.GRAPH_CONV_INIT = "normal"
    # Mesh sampling
    # loss weights
    # coord thresh
    cfg.MODEL.ROI_NOCS_HEAD.GT_COORD_THRESH = 0.0
    # Init ico_sphere level (only for when voxel_on is false)
    cfg.MODEL.ROI_NOCS_HEAD.ICO_SPHERE_LEVEL = -1
    cfg.MODEL.ROI_NOCS_HEAD.NUM_CONV = 0

    cfg.MODEL.ROI_NOCS_HEAD.NUM_BINS = 0
    cfg.MODEL.ROI_NOCS_HEAD.NORM = ""
    cfg.MODEL.ROI_NOCS_HEAD.NUM_CLASSES = 0


    cfg.MODEL.ROI_NOCS_HEAD.USE_SYMMETRY_LOSS =1
    cfg.MODEL.ROI_NOCS_HEAD.COORD_USE_BINS =1
    cfg.MODEL.ROI_NOCS_HEAD.COORD_SHARE_WEIGHTS =0
    cfg.MODEL.ROI_NOCS_HEAD.COORD_USE_DELTA =0
    cfg.MODEL.ROI_NOCS_HEAD.COORD_REGRESS_LOSS ='Soft_L1'
    cfg.MODEL.ROI_NOCS_HEAD.USE_BN = 1
    return cfg