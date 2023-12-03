import cv2
import datetime
import glob
import time
import numpy as np
import utilscopy
import _pickle as cPickle
import os
from detectron2.config import get_cfg
from detectron2.engine import(
    DefaultPredictor
)
import scipy.ndimage
import copy
from PIL import Image
from nocsrcnn.config import get_nocsrcnn_cfg_defaults

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
    assert instance_ids[-1] == 255
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
                print(bbox_file)
                scale_factor[i, :] = np.loadtxt(bbox_file)

            scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])
            #获得bbox三维的相对长度

        else:
            bbox_file = os.path.join(OBJ_MODEL_DIR, subset, words[2], words[3], 'bbox.txt')
            bbox = np.loadtxt(bbox_file)
            scale_factor[i, :] = bbox[0, :] - bbox[1, :]
            #但是发现bbox[0, :]与bbox[1, :]是相反数
            #又发现已经归一化好了
    
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
        scales[i, :] = scale_factor[inst_id - 1, :]
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

def load_mask(dataset_dict,subset):
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
        coord_map = coord_map[:, :, (2, 1, 0)]               
        #print('mask_in',mask_im.shape)
        masks, coords, class_ids, scales = process_data(mask_im, coord_map, dataset_dict["inst_dict"], meta_path,subset = subset)


    else:
        assert False

    return masks, coords, class_ids, scales, domain_label

def load_depth(dataset_dict):
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file.
    """
    if dataset_dict["source"] in ["CAMERA", "Real"]:
        depth_path = dataset_dict["image_path"] + '_depth.png'
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            # This is encoded depth image, let's convert
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'
    else:
        depth16 = None
    return depth16
    


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

def load_camera_scenes( dataset_dir, subset,if_calculate_mean=False):
    """Load a subset of the CAMERA dataset.
    dataset_dir: The root directory of the CAMERA dataset.
    subset: What to load (train, val)
    if_calculate_mean: if calculate the mean color of the images in this dataset
    """
    print('begin load camera')
    #f = open('/home/qiweili/detectron2/train_camera_0.out', 'r')
    #a = f.readlines()
    #for i in range(0, len(a)):
    #    a[i] = a[i].rstrip('\n')
    #    a[i]=int(a[i])
    #f.close()
    image_dir = os.path.join(dataset_dir, subset)
    source = "CAMERA"
    #print(image_dir)
    #image_dir = []
    #print('***********')
    folder_list = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]

    num_total_folders = len(folder_list)

    image_ids = range(int(num_total_folders *10))
    image_ids = range(20)
    color_mean = np.zeros((0, 3), dtype=np.float32)
    # Add images
    list = []
    j = 0
    for i in image_ids:

        image_id = int(i) % 10
        folder_id = int(i) // 10
        #image_id = 1
        #folder_id = 1
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
                            'inst_dict':inst_dict,'subset':subset})
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
    for folder in folder_list[:10]:
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
                            'inst_dict':inst_dict,'subset':subset})
            image_id += 1
    return list
 
def my_dataset_function(data):
    camera_dir = os.path.join('/data2', 'qiweili','camera')
    real_dir = os.path.join('/data2', 'qiweili','real')
    if data == 'val' :
        list_camera = load_camera_scenes(camera_dir,'val')
        return list_camera
    else :
        list_real = load_real_scenes(real_dir,'test')
        return list_real
    

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
    # for depth map
    if len(mask.shape) == 2:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale], order=0)
        new_padding = padding
    # for instance mask
    elif len(mask.shape) == 3:
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

def mapper(dataset_dict):
    #print(dataset_dict)


    #time.sleep(1)
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image_dir = dataset_dict["image_path"]
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
   
    mask, coord, class_ids, scales, domain_label = load_mask(dataset_dict,dataset_dict["subset"])
    depth=load_depth(dataset_dict)
    #image, window, scale, padding = resize_image(
    #    image, 
    #    min_dim=480, 
    #    max_dim=640,
    #    padding=1)
    #mask  = resize_mask(mask.transpose(1,2,0), scale, padding).transpose(2, 0, 1)
    boxes = extract_bboxes(mask)
    #oord = resize_mask(coord, scale, padding)
    #depth_ex = np.expand_dims(depth,2).repeat(2,axis=2)
    #depth = resize_mask(depth_ex, scale, padding)[:,:,0]
    #print(BitMasks(mask).get_bounding_boxes())
    #print('--------------')
    #instances.scales = scales
    return image , depth,mask,boxes,coord,class_ids,scales,domain_label

def max_min_coord(coord):
    print(np.max(coord[:,:,0]),np.max(coord[:,:,1]),np.max(coord[:,:,2]))
    print(np.min(coord[:,:,0]),np.min(coord[:,:,1]),np.min(coord[:,:,2]))
if __name__ == '__main__':




    camera_dir = os.path.join('/data2', 'qiweili','camera')
    real_dir = os.path.join('/data2', 'qiweili','real')
    coco_dir = os.path.join('/data2', 'qiweili','coco')
    

    #  real classes
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

    
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]

    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }



    mode = 'detect'
    #mode = 'eval'
    data = 'val'
    #data = 'real_test'
    draw = 1
    num_eval = -1
    assert mode in ['detect', 'eval']
    if mode == 'detect':
        #os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        cfg = get_cfg()
        cfg = get_nocsrcnn_cfg_defaults(cfg)
        cfg.merge_from_file(
            "configs/NOCSrcnn.yaml"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.ROI_NOCS_HEAD.NUM_CLASSES = 6
        cfg.MODEL.WEIGHTS = os.path.join("/data2/qiweili/logs/nocs_bin4/step3", "model_final.pth")  # 我们进行训练的 权值文件  存放处

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85 # 设置一个阈值
        predictor = DefaultPredictor(cfg)
        # Recreate the model in inference mode
        
        
        gt_dir = os.path.join('/data2','qiweili','gts', data)
        
        

        # Load trained weights (fill in path to trained weights here)




        save_per_images = 10
        now = datetime.datetime.now()
        save_dir = os.path.join('output/try', "{}_{:%Y%m%dT%H%M}".format(data, now))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        log_file = os.path.join(save_dir, 'error_log.txt')
        f_log = open(log_file, 'w')

        if data in ['real_train', 'real_test']:
            intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        else: ## CAMERA data
            intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

        elapse_times = []

        data_dict = my_dataset_function(data)
        if mode != 'eval':
            for i, image_dict in enumerate(data_dict):

                print('*'*50)
                image_start = time.time()
                print('Image id: ', image_dict['image_id'])
                image_path = image_dict['image_path']
                print(image_path)


                # record results
                result = {}

                # loading ground truth
                image ,depth , gt_mask,gt_bbox, gt_coord, gt_class_ids, gt_scales, gt_domain_label \
                    = mapper( image_dict)
                #cv2.imwrite('coord.jpg',(gt_coord*255).astype("int"))
                gt_mask = gt_mask.transpose(1,2,0)
                #gt_coord = np.zeros(gt_coord.shape)
                '''
                print(image.shape)
                print(depth.shape)
                print(gt_mask.shape)
                print(gt_bbox.shape)
                print(gt_coord.shape)
                print(gt_class_ids.shape)
                print(gt_scales.shape)
                '''
                result['image_id'] = image_dict['image_id']
                result['image_path'] = image_path

                result['gt_class_ids'] = gt_class_ids
                result['gt_bboxes'] = gt_bbox

                result['gt_RTs'] = None            
                result['gt_scales'] = gt_scales
                
                image_path_parsing = image_path.split('/')
                gt_pkl_path = os.path.join(gt_dir, 'results_{}_{}_{}.pkl'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                print('gt_pkl_path:',gt_pkl_path)
                #if (os.path.exists(gt_pkl_path)):
                if 0:
                    with open(gt_pkl_path, 'rb') as f:
                        gt = cPickle.load(f)
                    result['gt_RTs'] = gt['gt_RTs']
                    if 'handle_visibility' in gt:
                        result['gt_handle_visibility'] = gt['handle_visibility']
                        assert len(gt['handle_visibility']) == len(gt_class_ids)
                        print('got handle visibiity.')
                    else: 
                        result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
                else:
                    # align gt coord with depth to get RT
                    if not data in ['coco_val', 'coco_train']:
                        if len(gt_class_ids) == 0:
                            print('No gt instance exsits in this image.')

                        print('\nAligning ground truth...')
                        start = time.time()
                        result['gt_RTs'], _, error_message, _ = utilscopy.align(gt_class_ids, 
                                                                         gt_mask, 
                                                                         gt_coord, 
                                                                         depth, 
                                                                         intrinsics, 
                                                                         synset_names, 
                                                                         image_path,
                                                                         save_dir+'/'+'{}_{}_{}_gt_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                        
                        print('New alignment takes {:03f}s.'.format(time.time() - start))

                        if len(error_message):
                            f_log.write(error_message)

                    result['gt_handle_visibility'] = np.ones_like(gt_class_ids)

                ## detection
                start = time.time()
                detect_result = predictor(image)
                #print(detect_result)
                rr = detect_result['instances']
                #print(rr.pred_coord.shape)  #[num_instance, 28, 28, 3]
                #print(image.shape)
                #print(rr.pred_masks.shape) #[num_instance, 480, 640]
                #print(np.unique(rr.pred_masks.cpu().numpy()))
                #print(rr.pred_boxes)
                

                elapsed = time.time() - start
                #print(rr.pred_boxes.tensor.int().cpu().numpy())
                print('\nDetection takes {:03f}s.'.format(elapsed))
                #print(rr)
                r = utilscopy.prase_instance(rr)

                result['pred_class_ids'] = r['class_ids']
                #print(r['class_ids'])
                result['pred_bboxes'] = r['rois']
                result['pred_RTs'] = None   
                result['pred_scores'] = r['scores']
                #print(r['class_ids'])
                #print('len(r[class_ids]',len(r['class_ids']))
 
                if len(r['class_ids']) == 0:
                    print('No instance is detected.')

                print('Aligning predictions...')
                start = time.time()
                result['pred_RTs'], result['pred_scales'], error_message, elapses =  utilscopy.align(r['class_ids'], 
                                                                                        r['masks'], 
                                                                                        r['coords'], 
                                                                                        depth, 
                                                                                        intrinsics, 
                                                                                        synset_names, 
                                                                                        image_path,if_norm=True)
                                                                
                 #save_dir+'/'+'{}_{}_{}_pred_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))

                #exit(0)
                print('New alignment takes {:03f}s.'.format(time.time() - start))
                elapse_times += elapses
                if len(error_message):
                    f_log.write(error_message)


                if draw:
                    draw_rgb = False
                    #print(type( r['rois'][0][0]))
                    utilscopy.draw_detections(image, save_dir, data, image_path_parsing[-2]+'_'+image_path_parsing[-1], intrinsics, synset_names, draw_rgb,
                                            gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                                            r['rois'], r['class_ids'], r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'])
                          

                path_parse = image_path.split('/')
                image_short_path = '_'.join(path_parse[-3:])

                save_path = os.path.join(save_dir, 'results_{}.pkl'.format(image_short_path))
                with open(save_path, 'wb') as f:
                    cPickle.dump(result, f)
                print('Results of image {} has been saved to {}.'.format(image_short_path, save_path))

                
                elapsed = time.time() - image_start
                print('Takes {} to finish this image.'.format(elapsed))
                print('Alignment average time: ', np.mean(np.array(elapse_times)))
                print('\n')
            
            f_log.close()


    else:
        log_dir = 'output/val_20210910T1141'
        print(os.path.join(log_dir, 'results_*.pkl'))
        result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
        result_pkl_list = sorted(result_pkl_list)[:num_eval]
        assert len(result_pkl_list)

        final_results = []
        for pkl_path in result_pkl_list:
            with open(pkl_path, 'rb') as f:
                result = cPickle.load(f)
                if not 'gt_handle_visibility' in result:
                    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                    print('can\'t find gt_handle_visibility in the pkl.')
                else:
                    assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(result['gt_handle_visibility'], result['gt_class_ids'])


            if type(result) is list:
                final_results += result
            elif type(result) is dict:
                final_results.append(result)
            else:
                assert False

        #aps = utils.compute_degree_cm_mAP(final_results, synset_names, log_dir,
        #                                                            degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
        #                                                            shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
        #                                                            iou_3d_thresholds=np.linspace(0, 1, 101),
        #                                                            iou_pose_thres=0.1,
        #                                                            use_matches_for_pose=True)
        aps = utilscopy.compute_iou_mAP(final_results, synset_names, iou_3d_thresholds=np.linspace(0, 1, 101),
                                                                    iou_pose_thres=0.1,
                                                                    use_matches_for_pose=True)
       


    
