from detectron2.structures.masks import BitMasks
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data import (
    MetadataCatalog)
from detectron2.data.build import (
    get_detection_dataset_dicts,
    _train_loader_from_config,
    _test_loader_from_config,
    build_batch_data_loader,
    trivial_batch_collator
)
import scipy.ndimage
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import EvalHook
from detectron2.structures import Instances
from detectron2.data.samplers import InferenceSampler, TrainingSampler
import torch
import copy
import os
import numpy as np
import cv2
import glob
import random
from PIL import Image
import utilscopy
from pycocotools.coco import COCO
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
        image = np.array(Image.fromarray(image).resize((round(h * scale), round(w * scale))))
        #image = scipy.misc.imresize(
            #image, (round(h * scale), round(w * scale)))
    # Need padding?
    
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        #print(image)
        image = np.pad(image, padding, mode='constant', constant_values=0)
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

    mask = np.pad(mask, new_padding, mode='constant', constant_values=0)

    return mask

def process_data(mask_im, coord_map, inst_dict, meta_path, load_RT=False,subset = 'train'):
    #返回的形式：
    """
    masks = np.zeros([num_instance,h, w], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)
    """
    #print('---begin process_data-------')
    #print('-------inst_dict:   ',inst_dict) 
    #{1: 3, 2: 1, 3: 0, 4: 1, 5: 3, 6: 1, 7: 6, 8: 0, 9: 0, 10: 0, 11: 3, 12: 6}
    # parsing mask
    cdata = mask_im
    cdata = np.array(cdata, dtype=np.int32)
    # cdata : 640*480 没有物体为255 有物体为物体的编号 e.g.[1,2,4,255]
    #用了unique的话一个图片中一个物体有多个的话会混淆？是因为做的不是语义分割所以不需要区别个体？
    #那bouding box的产生会有问题吗？
    # instance ids
    instance_ids = list(np.unique(cdata))
    instance_ids = sorted(instance_ids)

    # remove background
    #assert instance_ids[-1] == 255
    if instance_ids[-1] == 255 :
        del instance_ids[-1]
    #print('instance_ids',instance_ids)
    cdata[cdata==255] = -1
    
    assert(np.unique(cdata).shape[0] < 20) #

    num_instance = len(instance_ids)
    h, w = cdata.shape

    # flip z axis of coord map
 
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]
    

    masks = np.zeros([num_instance,h, w], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    scales = np.zeros([num_instance, 3], dtype=np.float32)

    with open(meta_path, 'r') as f:
        lines = f.readlines()
    # lines 为 m 行（一个图片m个物体），每行为一个物体
    '''
    scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
    OBJ_MODEL_DIR = os.path.join('/data2','qiweili', 'data', 'obj_models')
    
    #以下循环获得bbox三维的归一化长度
    for i, line in enumerate(lines):
        words = line[:-1].split(' ')
        
        if len(words) == 3:
            ## real scanned objs
            if words[2][-3:] == 'npz':
                npz_path = os.path.join(OBJ_MODEL_DIR, 'real_val', words[2])
                with np.load(npz_path) as npz_file:
                    scale_factor[i, :] = npz_file['scale']
            else:
                bbox_file = os.path.join(OBJ_MODEL_DIR, 'real_'+subset, words[2]+'.txt')
                scale_factor[i, :] = np.loadtxt(bbox_file)

            scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])
            #获得bbox三维的相对长度

        else:
            bbox_file = os.path.join(OBJ_MODEL_DIR, subset, words[2], words[3], 'bbox.txt')
            bbox = np.loadtxt(bbox_file)
            scale_factor[i, :] = bbox[0, :] - bbox[1, :]
            #但是发现bbox[0, :]与bbox[1, :]是相反数
            #又发现已经归一化好了
    '''
    i = 0

    # delete ids of background objects and non-existing objects 
    inst_id_to_be_deleted = []
    for inst_id in inst_dict.keys():
        if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
            inst_id_to_be_deleted.append(inst_id)
    for delete_id in inst_id_to_be_deleted:
        del inst_dict[delete_id]
    for inst_id in instance_ids:  # instance mask is one-indexed
        if not inst_id in inst_dict:
            continue
        inst_mask = np.equal(cdata, inst_id)
        #获得每类物体的掩码
        assert np.sum(inst_mask) > 0
        assert inst_dict[inst_id]
            
        masks[i,:, :] = inst_mask
        #下行利用已经处理过的mask，处理coords,使得每个coords只含有一个类的物体
        coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

        # class ids is also one-indexed
        class_ids[i] = inst_dict[inst_id]
        #scales[i, :] = scale_factor[inst_id - 1, :]
        i += 1

    # print('before: ', inst_dict)
    masks = masks[:i, :, :]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1)
    
    class_ids = class_ids[:i]
    scales = scales[:i]
    #print('---end process_data-------')
    return masks, coords, class_ids, scales

def cocoid_to_dataid(coco_id):
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]
    temp = {44:1,51:2,73:5,47:6}
    if coco_id in temp:
        return temp[coco_id]
    return False

def load_mask(dataset_dict):
    """Generate instance masks for the objects in the image with the given ID.
    """
    #masks, coords, class_ids, scales, domain_label = None, None, None, None, None

    if dataset_dict["source"] in ["CAMERA", "Real"]:
        domain_label = 0 ## has coordinate map loss
        image_dir = dataset_dict["image_path"]
        mask_path = image_dir + '_mask.png'
        coord_path = image_dir+ '_coord.png'
        meta_path = image_dir+ '_meta.txt'
        assert os.path.exists(mask_path), "{} is missing".format(mask_path)
        assert os.path.exists(coord_path), "{} is missing".format(coord_path)

        #print('-------inst_dict:   ',inst_dict) #{1: 3, 2: 1, 3: 0, 4: 1, 5: 3, 6: 1, 7: 6, 8: 0, 9: 0, 10: 0, 11: 3, 12: 6}
        # meta 文件的前两列
        mask_im = cv2.imread(mask_path)[:, :, 2] #本来就二维，第三个2的参数可以去掉
        coord_map = cv2.imread(coord_path)[:, :, :3]
        coord_map = coord_map [:, :, (2, 1, 0)]                     
        #print('mask_in',mask_im.shape)
        masks, coords, class_ids, scales = process_data(mask_im, coord_map, dataset_dict["inst_dict"], meta_path)


    elif dataset_dict["source"]=="coco":
        domain_label = 1 ## no coordinate map loss

        instance_masks = []
        class_ids = []
        annotations = dataset_dict["annotations"]
        #coco_names= MetadataCatalog.get("my_dataset").coco_names
        #class_map = MetadataCatalog.get("my_dataset").coco_class_map
        #synset_names =MetadataCatalog.get("my_dataset").synset_names
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            #class_id = self.map_source_class_id(
            #    "coco.{}".format(annotation['category_id']))"category_id"
            #print(annotation)
            #print(type(annotation))
            coco_id  = annotation['category_id']
            
            class_id  = cocoid_to_dataid(coco_id)

            #class_id = synset_names.index(class_map[coco_names[annotation["category_id"]]])
            if class_id:
                m = utilscopy.annToMask(annotation, dataset_dict["height"],
                                    dataset_dict["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            masks = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            # Call super class to return an empty mask
            masks = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
        
        # use zero arrays as coord map for COCO images
        coords = np.zeros(masks.shape+(3,), dtype=np.float32)
        masks = masks.transpose(2,0,1)
        scales = np.ones((len(class_ids),3), dtype=np.float32)
        #print('\nwithout augmented, masks shape: {}'.format(masks.shape))
    


    return masks, coords, class_ids, scales, domain_label

def aug_load_mask(image, dataset_dict ,rotate_degree) :
    if dataset_dict["source"] in ["CAMERA", "Real"]:
        domain_label = 0 ## has coordinate map loss

        image_dir = dataset_dict["image_path"]
        mask_path = image_dir + '_mask.png'
        coord_path = image_dir+ '_coord.png'
        meta_path = image_dir+ '_meta.txt'

        mask_im = cv2.imread(mask_path)[:, :, 2]
        coord_map = cv2.imread(coord_path)[:, :, :3]
        coord_map = coord_map[:, :, ::-1]

        image, mask_im, coord_map = utilscopy.rotate_and_crop_images(image, 
                                                                    masks=mask_im, 
                                                                    coords=coord_map, 
                                                                    rotate_degree=rotate_degree)
        #cv2.imwrite('image.jpg',image[:,:,::-1])
        masks, coords, class_ids, scales = process_data(mask_im, coord_map, dataset_dict["inst_dict"], meta_path)
    elif dataset_dict["source"]:
        domain_label = 1 ## no coordinate map loss
        instance_masks = []
        class_ids = []
        annotations = dataset_dict["annotations"]
        #coco_names= MetadataCatalog.get("my_dataset").coco_names
        #class_map = MetadataCatalog.get("my_dataset").coco_class_map
        #synset_names =MetadataCatalog.get("my_dataset").synset_names
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            #class_id = self.map_source_class_id(
            #    "coco.{}".format(annotation['category_id']))"category_id"
            #print(annotation)
            #print(type(annotation))
            coco_id  = annotation['category_id']
            
            class_id  = cocoid_to_dataid(coco_id)

            #class_id = synset_names.index(class_map[coco_names[annotation["category_id"]]])
            if class_id:
                m = utilscopy.annToMask(annotation, dataset_dict["height"],
                                    dataset_dict["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)


        masks = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        image, masks = utilscopy.rotate_and_crop_images(image, 
                                                        masks=masks, 
                                                        coords=None, 
                                                        rotate_degree=rotate_degree)
        if len(masks.shape)==2:
                masks = masks[:, :, np.newaxis]
        
        final_masks = []
        final_class_ids = []
        for i in range(masks.shape[-1]):
            m = masks[:, :, i]
            if m.max() < 1:
                continue
            final_masks.append(m)
            final_class_ids.append(class_ids[i])

        if final_class_ids:
            masks = np.stack(final_masks, axis=2)
            class_ids = np.array(final_class_ids, dtype=np.int32)
        else:
            # Call super class to return an empty mask
            masks = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
        
        
        coords = np.zeros(masks.shape+(3,), dtype=np.float32)
        masks = masks.transpose(2,0,1)
        scales = np.ones((len(class_ids),3), dtype=np.float32)
    return image ,masks, coords, class_ids, scales, domain_label

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        #对于一张照片中的不同类的物体分别处理
        # Bounding box.
        #返回mask中物体所在的行坐标
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        #返回mask中物体所在的列坐标
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

# use this dataloader instead of the default
def load_camera_scenes( dataset_dir, subset,if_calculate_mean=False):
    """Load a subset of the CAMERA dataset.
    dataset_dir: The root directory of the CAMERA dataset.
    subset: What to load (train, val)
    if_calculate_mean: if calculate the mean color of the images in this dataset
    """
    print('begin load camera')
    image_dir = os.path.join(dataset_dir, subset)
    source = "CAMERA"
    #print(image_dir)
    #image_dir = []
    #print('***********')
    folder_list = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]

    num_total_folders = len(folder_list)
    image_ids = range(int(num_total_folders *10))
    #image_ids = range(10)
    if subset == 'val' :
        image_ids = range(30)
    color_mean = np.zeros((0, 3), dtype=np.float32)
    # Add images
    list = []
    j = 0
    for i in image_ids:
        #if i in a :
        #    continue
        image_id = int(i) % 10
        folder_id = int(i) // 10
        #image_id = 1
        #folder_id = 0
        image_path = os.path.join(image_dir, '{:05d}'.format(folder_id), '{:04d}'.format(image_id))
        color_path = image_path + '_color.png'
        if not os.path.exists(color_path):
            continue
        
        meta_path = os.path.join(image_dir, '{:05d}'.format(folder_id), '{:04d}_meta.txt'.format(image_id))
        inst_dict = {}
        with open(meta_path, 'r') as f:
            for line in f:
                line_info = line.split(' ')
                inst_id = int(line_info[0])  ##one-indexed
                cls_id = int(line_info[1])  ##zero-indexed
                # skip background objs
                # symmetry_id = int(line_info[2])
                inst_dict[inst_id] = cls_id

        #width = cfg.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
        #height =cfg.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]
        
        list.append({'source':source ,'image_id':j,'image_path':image_path,
                            'inst_dict':inst_dict})
        j=j+1
    print(j)
    print('load camera successfully')
    return list

def load_real_scenes(dataset_dir, subset,if_calculate_mean=False):
    """Load a subset of the Real dataset.
    dataset_dir: The root directory of the Real dataset.
    subset: What to load (train, val, test)
    if_calculate_mean: if calculate the mean color of the images in this dataset
    """

    source = "Real"

    folder_name = 'train' if subset == 'train' else 'test'
    image_dir = os.path.join(dataset_dir, folder_name)
    
    #print(image_dir)
    #image_dir=[]
    #print('*****')

    folder_list = [name for name in glob.glob(image_dir + '/*') if os.path.isdir(name)]
    folder_list = sorted(folder_list)
    list = []
    image_id = 0
    for folder in folder_list:
        image_list = glob.glob(os.path.join(folder, '*_color.png'))
        image_list = sorted(image_list)

        for image_full_path in image_list:
            image_name = os.path.basename(image_full_path)
            image_ind = image_name.split('_')[0]
            image_path = os.path.join(folder, image_ind)
            
            meta_path = image_path + '_meta.txt'
            inst_dict = {}
            with open(meta_path, 'r') as f:
                for line in f:
                    line_info = line.split(' ')
                    inst_id = int(line_info[0])  ##one-indexed
                    cls_id = int(line_info[1])  ##zero-indexed
                    # symmetry_id = int(line_info[2])
                    inst_dict[inst_id] = cls_id

            
            #width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
            #height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]
            list.append({'source':source ,'image_id':image_id,'image_path':image_path,
                            'inst_dict':inst_dict})
            image_id += 1
    return list

def load_coco(dataset_dir, subset = 'train')   :  
    """Load a subset of the COCO dataset.
    dataset_dir: The root directory of the COCO dataset.
    subset: What to load (train, val, minival, val35k)
    class_ids: If provided, only loads images that have the given classes.
    """
    source = "coco"


    image_dir = os.path.join(dataset_dir, "images", "train2017" if subset == "train"
    else "val2017")
    #print('************')
    #print(image_dir)
    # Create COCO object
    json_path_dict = {
        "train": "annotations/instances_train2017.json",
        "val": "annotations/instances_val2017.json",
    }
    coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))

    # Load all classes or a subset?
    class_names = MetadataCatalog.get("coco_train_dataset").coco_class_map
    image_ids = set()
    class_ids = coco.getCatIds(catNms=class_names)

    for cls_name in class_names:
        catIds = coco.getCatIds(catNms=[cls_name])
        imgIds = coco.getImgIds(catIds=catIds )
        image_ids = image_ids.union(set(imgIds))

    image_ids = list(set(image_ids))
    print(len(image_ids))
    '''
    这里需要add class
    # Add classes
    for cls_id in class_ids:
        self.add_class("coco", cls_id, coco.loadCats(cls_id)[0]["name"])
        print('Add coco class: '+coco.loadCats(cls_id)[0]["name"])
    '''
    # Add images
    list_out = []
    for i, image_id in enumerate(image_ids):
        list_out.append({'source':source ,'image_id':image_id,
                'image_path':os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                'width':coco.imgs[image_id]["width"],
                 'height':coco.imgs[image_id]["height"],
                 'annotations':coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))})
    
    return list_out
    
def camera_val_function():
    
    camera_dir = os.path.join('/data2', 'qiweili','camera')
    list_camera = load_camera_scenes(camera_dir,'val')
    return list_camera

def camera_train_function():
    
    camera_dir = os.path.join('/data2', 'qiweili','camera')
    list_camera = load_camera_scenes(camera_dir,'train')
    return list_camera

def real_train_function():

    real_dir = os.path.join('/data2', 'qiweili','real')
    list_real = load_real_scenes(real_dir,'train')
    return list_real

def coco_train_function():

    coco_dir = os.path.join('/data2', 'qiweili','coco')
    list_coco = load_coco(coco_dir,"train")
    return list_coco

 # Show how to implement a minimal mapper, similar to the default DatasetMapper

j = 0

def mapper(dataset_dict):
    #print(dataset_dict)

    #time.sleep(1)
    #print('use mapper')
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image_dir = dataset_dict["image_path"]
    #print(image_dir)
    #print(image_dir)
    source = dataset_dict["source"]
    if source in ["CAMERA", "Real"]:
        image_path = image_dir + '_color.png'
        assert os.path.exists(image_path), "{} is missing".format(image_path)

        #depth_path = info["path"] + '_depth.png'
    elif dataset_dict["source"]=='coco':
        image_path = image_dir
        #image_path = '/data2/qiweili/coco/images/train2017/000000522232.jpg'

    else:
        assert False, "[ Error ]: Unknown image path: {}".format(image_dir)
    image = cv2.imread(image_path)[:, :, :3]
    
    image = image[:, :, ::-1]
    #image_d = image[:, :, ::-1].copy()
    # If grayscale. Convert to RGB for consistency.
    #cv2.imwrite('origin.jpg',image)
    if image.ndim != 3:
        image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    rotate_degree = np.random.uniform(5, 15)
    image_after = image
    aug = True
    if aug :
        image_after,mask, coord, class_ids, scales, domain_label = aug_load_mask(image,dataset_dict,rotate_degree)
    else :     
        mask, coord, class_ids, scales, domain_label = load_mask(dataset_dict)
    #画出aug前,没有reszie的图
    #mask_before, coord_, class_ids_, scales_, domain_label_ = load_mask(dataset_dict)
    
    class_ids = class_ids -1
    #mask_before=mask
    #bbox_before = extract_bboxes(mask_before)
    image =np.ascontiguousarray(image)
   
    
    image_after, window, scale, padding = resize_image(
        image_after, 
        min_dim=480, 
        max_dim=640,
        padding=1)
    #上需要改动
    #print(image.shape)
    mask  = resize_mask(mask.transpose(1,2,0), scale, padding).transpose(2, 0, 1)
    #if(mask.shape[0]==0) :
    #    exit(0)

    coord = resize_mask(coord, scale, padding)

    image = image_after
    image=torch.from_numpy(image.transpose(2, 0, 1))
    
    instances = Instances(tuple([image.shape[1],image.shape[2]]))
    coord = torch.from_numpy(coord.transpose(2,0,1,3))
    instances.set("gt_domain_label",torch.from_numpy(np.array([domain_label]*len(class_ids))))

    instances.set("gt_coord",coord)
    instances.gt_classes = torch.from_numpy(class_ids).long()
    instances.gt_boxes = BitMasks(mask).get_bounding_boxes()
    #print(Boxes(bbox))
    
    #print('mask',mask.shape)
    #print('--------------')
    instances.gt_masks =BitMasks(mask)
    if len(class_ids) == 0:
        return None
    #print(BitMasks(mask).get_bounding_boxes())
    #print('--------------')
    #instances.scales = scales
    return {
       # create the format that the model expects
        "image": image,
        "height" :640,
        "width":640  ,
        "instances" : instances ,
        #"coord" :coord,
        #"scales" :scales
    }

class MyMapDataset(MapDataset) :
    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data
            else :
                l=len(self._dataset)
                i =random.randint(1,l)
                print(l)
                cur_idx = (cur_idx+i)%l 
                print(cur_idx)
                #input()
                continue

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
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

class mydataloader :
    def __init__(self,cfg):
        self.cfg = cfg   
        self.loader = []  
        def fn (dataset):
            return   get_detection_dataset_dicts(
                dataset,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
        dataset = fn(cfg.DATASETS.TRAIN[0])
        self.loader.append( iter(build_detection_train_loader(cfg, mapper=mapper,dataset=dataset)) )
        dataset = fn(cfg.DATASETS.TRAIN[1])
        self.loader.append( iter(build_detection_train_loader(cfg, mapper=mapper,dataset=dataset)) )
        dataset = fn(cfg.DATASETS.TRAIN[2])
        self.loader.append( iter(build_detection_train_loader(cfg, mapper=mapper,dataset=dataset)) )

    def __next__(self):   #__next__用于返回下一个，返回下一个才能被称之为迭代器。
        weight = self.cfg.DATASETS.WEIGHT
        weight = weight/np.sum(weight)
        assert np.sum(weight) == 1, "[ Error ]: total sum of weights is {} != 1".format(np.sum(weight))
        source_ind = np.random.choice([0,1,2], 1, p=weight)[0]
        return next(self.loader[source_ind])

    def __iter__(self):   #__iter__用于返回自身，返回自身才能被调用。
        return self

class Trainer(DefaultTrainer):

    def build_train_loader(cls, cfg):
        return mydataloader(cfg)
    @classmethod
    def mytest(cls, cfg, model,dataset_name):
        #model.eval()
        loss = {}
        re ={}
        with torch.no_grad():
            l = len( dataset_name)
            for i in dataset_name :
                a = model(i)
                if not loss :
                    loss = a
                else :
                    for key ,value in a.items():
                        loss[key]=loss[key]+value
            for key ,value in loss.items():
                re['val_'+key] = loss[key].item()/l
            #print(re)
        model.train()
        return re

from nocsrcnn.modeling.roi_heads import NOCSRCNNROIHeads
from nocsrcnn.config import get_nocsrcnn_cfg_defaults


def set_coco():
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]
    MetadataCatalog.get("coco_train_dataset").set(coco_names= coco_names)
    MetadataCatalog.get("coco_train_dataset").set(coco_class_map= class_map)
    MetadataCatalog.get("coco_train_dataset").set(synset_names = synset_names)
    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    MetadataCatalog.get("coco_train_dataset").set(coco_cls_ids=coco_cls_ids)



if __name__ == "__main__":

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    cfg = get_cfg()
    cfg = get_nocsrcnn_cfg_defaults(cfg)
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.merge_from_file(
        "configs/NOCSrcnn.yaml"
        #"/home/qiweili/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yasml"
    )
    set_coco()
    #cfg.DATALOADER.NUM_WORKERS = 0
    DatasetCatalog.register("camera_train_dataset", camera_train_function)
    DatasetCatalog.register("real_train_dataset", real_train_function)
    DatasetCatalog.register("coco_train_dataset", coco_train_function)
    DatasetCatalog.register("camera_val_dataset", camera_val_function)
    cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs", "model_final_280758.pkl") 
    #cfg.MODEL.WEIGHTS=''
    cfg.SOLVER.BASE_LR = 0.001
    cfg.DATASETS.TRAIN = ("camera_train_dataset","real_train_dataset","coco_train_dataset")
    cfg.DATASETS.TEST = ("camera_val_dataset",)
    cfg.DATASETS.WEIGHT = [3,1,1]
    
    #os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.STEPS = (200, 300)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.ROI_NOCS_HEAD.NUM_CLASSES = 6
    

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    
    cfg.OUTPUT_DIR = '/data2/qiweili/logs/nocs_bin4/step1'
    cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs", "model_final_280758.pkl") 
    cfg.MODEL.BACKBONE.FREEZE_AT = 6
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = (75000, 85000)
    trainer = Trainer(cfg)
    model  =  trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(cfg, mapper=mapper,dataset_name=cfg.DATASETS.TEST)
    trainer.register_hooks([EvalHook(30,lambda:trainer.mytest(cfg,trainer.model,val_dataloader))])
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    cfg.OUTPUT_DIR = '/data2/qiweili/logs/nocs_bin4/step2'
    cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs/nocs_bin4/step1", "model_final.pth") 
    cfg.MODEL.BACKBONE.FREEZE_AT = 4
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / 10.
    cfg.SOLVER.MAX_ITER = 130000
    cfg.SOLVER.STEPS = (105000, 115000)
    trainer = Trainer(cfg)
    model  =  trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(cfg, mapper=mapper,dataset_name=cfg.DATASETS.TEST)
    trainer.register_hooks([EvalHook(30,lambda:trainer.mytest(cfg,trainer.model,val_dataloader))])
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.OUTPUT_DIR = '/data2/qiweili/logs/nocs_bin4/step3'
    cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs/nocs_bin4/step2", "model_final.pth") 
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / 10.
    cfg.SOLVER.MAX_ITER = 400000
    cfg.SOLVER.STEPS = (300000, 330000)
    trainer = Trainer(cfg)
    model  =  trainer.build_model(cfg)
    val_dataloader = build_detection_test_loader(cfg, mapper=mapper,dataset_name=cfg.DATASETS.TEST)
    trainer.register_hooks([EvalHook(30,lambda:trainer.mytest(cfg,trainer.model,val_dataloader))])
    trainer.resume_or_load(resume=False)
    trainer.train()
