from ipdb import set_trace
import torch
import cutoop
from cutoop.image_meta import ImageMetaData
from detectron2.structures.masks import BitMasks
from detectron2.config import get_cfg
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.build import (
    get_detection_dataset_dicts,
    _train_loader_from_config,
    _test_loader_from_config,
    build_batch_data_loader,
    trivial_batch_collator,
)
import scipy.ndimage
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import EvalHook
from detectron2.structures import Instances
from detectron2.data.samplers import InferenceSampler, TrainingSampler
import copy
import os
import numpy as np
import random
from PIL import Image
import utilscopy
from detectron2.config import configurable


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = np.array(
            Image.fromarray(image).resize((round(h * scale), round(w * scale)))
        )
        # image = scipy.misc.imresize(
        # image, (round(h * scale), round(w * scale)))
    # Need padding?

    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        # print(image)
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image, the mask, and the coordinate map are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    # for instance mask
    if len(mask.shape) == 3:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        new_padding = padding
    # for coordinate map
    elif len(mask.shape) == 4:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1, 1], order=0)
        new_padding = padding + [(0, 0)]
    else:
        assert False

    mask = np.pad(mask, new_padding, mode="constant", constant_values=0)

    return mask


def process_data(mask_im, coord_map, meta: ImageMetaData, image_path: str):
    # 返回的形式：
    """
    masks = np.zeros([num_instance,h, w], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)
    """
    # print('---begin process_data-------')
    # print('-------inst_dict:   ',inst_dict)
    # {1: 3, 2: 1, 3: 0, 4: 1, 5: 3, 6: 1, 7: 6, 8: 0, 9: 0, 10: 0, 11: 3, 12: 6}
    # parsing mask
    cdata = mask_im
    cdata = np.array(cdata, dtype=np.int32)
    # cdata : 640*480 没有物体为255 有物体为物体的编号 e.g.[1,2,4,255]
    # 用了unique的话一个图片中一个物体有多个的话会混淆？是因为做的不是语义分割所以不需要区别个体？
    # 那bouding box的产生会有问题吗？
    # instance ids

    # after random croppying, the object might be invisiable
    occur_ids = np.unique(cdata)
    instance_ids = [
        obj.mask_id for obj in meta.objects if obj.is_valid and obj.mask_id in occur_ids
    ]
    instance_ids = sorted(instance_ids)

    # # remove background
    # # assert instance_ids[-1] == 255
    # if instance_ids[-1] == 255:
    #     del instance_ids[-1]
    # if instance_ids[0] == 0:
    #     del instance_ids[0]
    # # print('instance_ids',instance_ids)
    cdata[cdata == 255] = -1
    cdata[cdata == 0] = -1

    # FIXME
    assert np.unique(cdata).shape[0] < 30  #

    num_instance = len(instance_ids)
    h, w = cdata.shape

    masks = np.zeros([num_instance, h, w], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)

    # # lines 为 m 行（一个图片m个物体），每行为一个物体
    # """
    # scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
    # OBJ_MODEL_DIR = os.path.join('/data2','qiweili', 'data', 'obj_models')

    # #以下循环获得bbox三维的归一化长度
    # for i, line in enumerate(lines):
    #     words = line[:-1].split(' ')

    #     if len(words) == 3:
    #         ## real scanned objs
    #         if words[2][-3:] == 'npz':
    #             npz_path = os.path.join(OBJ_MODEL_DIR, 'real_val', words[2])
    #             with np.load(npz_path) as npz_file:
    #                 scale_factor[i, :] = npz_file['scale']
    #         else:
    #             bbox_file = os.path.join(OBJ_MODEL_DIR, 'real_'+subset, words[2]+'.txt')
    #             scale_factor[i, :] = np.loadtxt(bbox_file)

    #         scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])
    #         #获得bbox三维的相对长度

    #     else:
    #         bbox_file = os.path.join(OBJ_MODEL_DIR, subset, words[2], words[3], 'bbox.txt')
    #         bbox = np.loadtxt(bbox_file)
    #         scale_factor[i, :] = bbox[0, :] - bbox[1, :]
    #         #但是发现bbox[0, :]与bbox[1, :]是相反数
    #         #又发现已经归一化好了
    # """

    for obj in meta.objects:
        if not obj.mask_id in instance_ids:
            continue
        i = instance_ids.index(obj.mask_id)
        inst_mask = cdata == obj.mask_id
        assert np.sum(inst_mask) > 0, f"invalid obj: {obj}, {image_path}"
        masks[i, :, :] = inst_mask
        # 下行利用已经处理过的mask，处理coords,使得每个coords只含有一个类的物体
        coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

        class_ids[i] = obj.meta.class_label

    # delete ids of background objects and non-existing objects
    # i = 0
    # inst_id_to_be_deleted = []
    # for inst_id in inst_dict.keys():
    #     if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
    #         inst_id_to_be_deleted.append(inst_id)
    # for delete_id in inst_id_to_be_deleted:
    #     del inst_dict[delete_id]
    # for inst_id in instance_ids:  # instance mask is one-indexed
    #     if not inst_id in inst_dict:
    #         continue
    #     inst_mask = np.equal(cdata, inst_id)
    #     # 获得每类物体的掩码
    #     assert np.sum(inst_mask) > 0
    #     assert inst_dict[inst_id]

    #     masks[i, :, :] = inst_mask
    #     # 下行利用已经处理过的mask，处理coords,使得每个coords只含有一个类的物体
    #     coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

    #     # class ids is also one-indexed
    #     class_ids[i] = inst_dict[inst_id]
    #     # scales[i, :] = scale_factor[inst_id - 1, :]
    #     i += 1

    # print('before: ', inst_dict)
    # masks = masks[:i, :, :]
    # coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1)

    # class_ids = class_ids[:i]
    # scales = scales[:i]
    # print('---end process_data-------')
    return masks, coords, class_ids, None  # scales


def load_mask(dataset_dict):
    """Generate instance masks for the objects in the image with the given ID."""
    # masks, coords, class_ids, scales, domain_label = None, None, None, None, None

    domain_label = 0  ## has coordinate map loss

    mask_path = dataset_dict["mask_image"]
    coord_path = dataset_dict["coord_image"]
    meta: ImageMetaData = dataset_dict["metadata"]
    assert os.path.exists(mask_path), "{} is missing".format(mask_path)
    assert os.path.exists(coord_path), "{} is missing".format(coord_path)

    # print('-------inst_dict:   ',inst_dict) #{1: 3, 2: 1, 3: 0, 4: 1, 5: 3, 6: 1, 7: 6, 8: 0, 9: 0, 10: 0, 11: 3, 12: 6}
    # meta 文件的前两列
    mask_im = cutoop.data_loader.Dataset.load_mask(mask_path)
    coord_map = cutoop.data_loader.Dataset.load_coord(coord_path)
    # mask_im = cv2.imread(mask_path)[:, :, 2]  # 本来就二维，第三个2的参数可以去掉
    # coord_map = cv2.imread(coord_path)[:, :, :3]
    # coord_map = coord_map[:, :, (2, 1, 0)]
    # print('mask_in',mask_im.shape)
    masks, coords, class_ids, _ = process_data(
        mask_im, coord_map, meta, dataset_dict["file_name"]
    )

    return masks, coords, class_ids, None, domain_label


def aug_load_mask(image, dataset_dict, rotate_degree):
    domain_label = 0  ## has coordinate map loss

    mask_path = dataset_dict["mask_image"]
    coord_path = dataset_dict["coord_image"]
    meta: ImageMetaData = dataset_dict["metadata"]
    # image_dir = dataset_dict["file_name"]
    # meta_path = image_dir + "_meta.txt"

    mask_im = cutoop.data_loader.Dataset.load_mask(mask_path)
    coord_map = cutoop.data_loader.Dataset.load_coord(coord_path)
    # coord_map = coord_map[:, :, ::-1]

    image, mask_im, coord_map = utilscopy.rotate_and_crop_images(
        image, masks=mask_im, coords=coord_map, rotate_degree=rotate_degree
    )
    # cv2.imwrite('image.jpg',image[:,:,::-1])
    masks, coords, class_ids, _ = process_data(
        mask_im, coord_map, meta, dataset_dict["file_name"]
    )
    return image, masks, coords, class_ids, None, domain_label


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # 对于一张照片中的不同类的物体分别处理
        # Bounding box.
        # 返回mask中物体所在的行坐标
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        # 返回mask中物体所在的列坐标
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


# Show how to implement a minimal mapper, similar to the default DatasetMapper


def mapper(dataset_dict):
    # print(dataset_dict)

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image_path = dataset_dict["file_name"]
    assert os.path.exists(image_path), "{} is missing".format(image_path)
    image = cutoop.data_loader.Dataset.load_color(image_path)
    assert image.ndim == 3

    # image_d = image[:, :, ::-1].copy()
    # If grayscale. Convert to RGB for consistency.
    # cv2.imwrite('origin.jpg',image)

    rotate_degree = np.random.uniform(5, 15)
    image_after = image
    aug = True
    if aug:
        image_after, mask, coord, class_ids, _, domain_label = aug_load_mask(
            image, dataset_dict, rotate_degree
        )
    else:
        mask, coord, class_ids, _, domain_label = load_mask(dataset_dict)
    # 画出aug前,没有reszie的图
    # mask_before, coord_, class_ids_, scales_, domain_label_ = load_mask(dataset_dict)

    class_ids = class_ids - 1
    assert min(class_ids) >= 0
    # mask_before=mask
    # bbox_before = extract_bboxes(mask_before)
    image = np.ascontiguousarray(image)

    image_after, _, scale, padding = resize_image(
        image_after, min_dim=480, max_dim=640, padding=1
    )
    # 上需要改动
    # print(image.shape)
    mask = resize_mask(mask.transpose(1, 2, 0), scale, padding).transpose(2, 0, 1)
    # if(mask.shape[0]==0) :
    #    exit(0)

    coord = resize_mask(coord, scale, padding)

    image = image_after
    image = torch.from_numpy(image.transpose(2, 0, 1))

    instances = Instances(tuple([image.shape[1], image.shape[2]]))
    coord = torch.from_numpy(coord.transpose(2, 0, 1, 3))
    instances.set(
        "gt_domain_label", torch.from_numpy(np.array([domain_label] * len(class_ids)))
    )

    instances.set("gt_coord", coord)
    instances.gt_classes = torch.from_numpy(class_ids).long()
    instances.gt_boxes = BitMasks(mask).get_bounding_boxes()
    # print(Boxes(bbox))

    # print('mask',mask.shape)
    # print('--------------')
    instances.gt_masks = BitMasks(mask)
    if len(class_ids) == 0:
        return None
    # print(BitMasks(mask).get_bounding_boxes())
    # print('--------------')
    # instances.scales = scales
    return {
        # create the format that the model expects
        "image": image,
        "height": 640,
        "width": 640,
        "instances": instances,
        # "coord" :coord,
        # "scales" :scales
    }


class MyMapDataset(MapDataset):
    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data
            else:
                l = len(self._dataset)
                i = random.randint(1, l)
                print(l)
                cur_idx = (cur_idx + i) % l
                print(cur_idx)
                # input()
                continue


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
):

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MyMapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0):

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MyMapDataset(dataset, mapper)
    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


class mydataloader:
    def __init__(self, cfg):
        self.cfg = cfg

        def fn(dataset):
            return get_detection_dataset_dicts(
                dataset,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=(
                    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                    if cfg.MODEL.KEYPOINT_ON
                    else 0
                ),
                proposal_files=(
                    cfg.DATASETS.PROPOSAL_FILES_TRAIN
                    if cfg.MODEL.LOAD_PROPOSALS
                    else None
                ),
            )

        dataset = fn(cfg.DATASETS.TRAIN[0])
        self.loader = iter(
            build_detection_train_loader(cfg, mapper=mapper, dataset=dataset)
        )

    def __next__(self):  # __next__用于返回下一个，返回下一个才能被称之为迭代器。
        return next(self.loader)

    def __iter__(self):  # __iter__用于返回自身，返回自身才能被调用。
        return self


class Trainer(DefaultTrainer):
    def build_train_loader(cls, cfg):
        return mydataloader(cfg)

    @classmethod
    def mytest(cls, cfg, model, dataset_name):
        # model.eval()
        loss = {}
        re = {}
        with torch.no_grad():
            l = len(dataset_name)
            for i in dataset_name:
                a = model(i)
                if not loss:
                    loss = a
                else:
                    for key, value in a.items():
                        loss[key] = loss[key] + value
            for key, value in loss.items():
                re["val_" + key] = loss[key].item() / l
            # print(re)
        model.train()
        return re


from nocsrcnn.modeling.roi_heads import NOCSRCNNROIHeads
from nocsrcnn.config import get_nocsrcnn_cfg_defaults


if __name__ == "__main__":
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    cfg = get_cfg()
    cfg = get_nocsrcnn_cfg_defaults(cfg)
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.merge_from_file(
        "configs/NOCSrcnn.yaml"
        # "/home/qiweili/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yasml"
    )
    # cfg.DATALOADER.NUM_WORKERS = 0

    import register_oop as _

    # DatasetCatalog.register("camera_train_dataset", camera_train_function)
    # DatasetCatalog.register("real_train_dataset", real_train_function)
    # DatasetCatalog.register("coco_train_dataset", coco_train_function)
    # DatasetCatalog.register("camera_val_dataset", camera_val_function)
    # cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs", "model_final_280758.pkl")
    # cfg.MODEL.WEIGHTS=''
    cfg.SOLVER.BASE_LR = 0.001
    cfg.DATASETS.TRAIN = ("oop_train",)
    cfg.DATASETS.TEST = ("oop_test",)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.STEPS = (200, 300)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    cfg.OUTPUT_DIR = "output_step1"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs", "model_final_280758.pkl")
    cfg.MODEL.BACKBONE.FREEZE_AT = 6
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = (75000, 85000)
    print(cfg)
    trainer = Trainer(cfg)
    model = trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(
        cfg, mapper=mapper, dataset_name=cfg.DATASETS.TEST
    )
    trainer.register_hooks(
        [EvalHook(30, lambda: trainer.mytest(cfg, trainer.model, val_dataloader))]
    )
    trainer.resume_or_load(resume=False)
    set_trace()
    trainer.train()

    """
    cfg.OUTPUT_DIR = "output_step2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(
        "/data2/qiweili/logs/nocs_bin4/step1", "model_final.pth"
    )
    cfg.MODEL.BACKBONE.FREEZE_AT = 4
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / 10.0
    cfg.SOLVER.MAX_ITER = 130000
    cfg.SOLVER.STEPS = (105000, 115000)
    trainer = Trainer(cfg)
    model = trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(
        cfg, mapper=mapper, dataset_name=cfg.DATASETS.TEST
    )
    trainer.register_hooks(
        [EvalHook(30, lambda: trainer.mytest(cfg, trainer.model, val_dataloader))]
    )
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.OUTPUT_DIR = "output_step3"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(
        "/data2/qiweili/logs/nocs_bin4/step2", "model_final.pth"
    )
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / 10.0
    cfg.SOLVER.MAX_ITER = 400000
    cfg.SOLVER.STEPS = (300000, 330000)
    trainer = Trainer(cfg)
    model = trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(
        cfg, mapper=mapper, dataset_name=cfg.DATASETS.TEST
    )
    trainer.register_hooks(
        [EvalHook(30, lambda: trainer.mytest(cfg, trainer.model, val_dataloader))]
    )
    trainer.resume_or_load(resume=False)
    trainer.train()
    """
