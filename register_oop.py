"""
Import this script to register the OmniObjectPose dataset under the detectron2 framework.

To run this script you need to:

- install `cutoop` package by executing `pip install .` under `OmniObjectPose/common`.
- place `OmniObjectPose/data_generation/configs/obj_meta.json` at the same folder of this file.
- change the `ROOT` variable below to the data folder.

It registers "oop_train" and "oop_test" for training and testing data, respectively.
"""

ROOT = "/root/autodl-tmp/OmniObjectPose/render_v1_down/03/render/v1/03/"

import glob
import os

import cv2
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from cutoop.image_meta import ImageMetaData
from cutoop.obj_meta import ObjectMetaData
from cutoop.data_loader import Dataset
from cutoop.utils import pool_map

from detectron2.utils.visualizer import Visualizer


def glob_prefix(root: str):
    prefs = [
        path[:-9]
        for path in glob.glob(os.path.join(root, "**/*_color.png"), recursive=True)
    ]
    return prefs


def get_record(args):
    id, pre = args
    filename = pre + "color.png"
    assert os.path.exists(filename), f"{filename} not exist"
    image = Dataset.load_color(filename)
    h, w = image.shape[:2]
    metadata = ImageMetaData.load_json(pre + "meta.json")
    # masks = Dataset.load_mask(pre + "mask.exr")

    # annotations = []
    # for obj in metadata.objects:
    #     ys, xs = tuple(np.argwhere(masks == obj.mask_id).T)
    #     poly = [(x + 0.5, y + 0.5) for x, y in zip(xs, ys)]

    #     if len(poly) < 4: # ignore this object
    #         continue

    #     annotations.append(
    #         {
    #             "category_id": obj.meta.class_label,
    #             "bbox": [np.min(xs), np.min(ys), np.max(xs), np.max(ys)],
    #             "bbox_mode": BoxMode.XYXY_ABS,
    #             "segmentation": [poly],
    #         }
    #     )

    record = {
        "file_name": filename,
        "height": h,
        "width": w,
        "image_id": id,
        "metadata": metadata,
        "mask_image": pre + "mask.exr",
        "coord_image": pre + "coord.png",
        # "annotations": annotations,
    }
    return record


def get_oop_dicts(root: str):
    prefixes = glob_prefix(root)
    print(f"get oop dicts {root}")
    dataset_dicts = pool_map(get_record, list(enumerate(prefixes)), processes=32)
    # import pickle
    # with open("oop.pkl", "wb") as f:
    #     pickle.dump(dataset_dicts, f)
    if "test" in root:
        dataset_dicts = dataset_dicts[:50]
    return dataset_dicts


objmeta: ObjectMetaData = ObjectMetaData.load_json(
    os.path.join(os.path.dirname(__file__), "obj_meta.json")
)

# register oop_train and oop_test
for d in ["train", "test"]:
    # do not access temporary variable in closure
    DatasetCatalog.register(
        "oop_" + d,
        lambda d=d: get_oop_dicts(os.path.join(ROOT, d)),
    )
    assert min(map(lambda x: x.label, objmeta.class_list)) > 0
    MetadataCatalog.get("oop_" + d).set(
        thing_classes=["background"] + [c.name for c in objmeta.class_list],
        # Used by all instance detection/segmentation tasks in the COCO format.
        # A mapping from instance class ids in the dataset to contiguous ids in range [0, #class).
        thing_dataset_id_to_contiguous_id={
            c.label: c.label - 1 for c in objmeta.class_list
        },
        evaluator_type="oop",
        image_root=os.path.join(ROOT, d) + d,
        # sem_seg_root=gt_dir,
        ignore_label=255,
    )

if __name__ == "__main__":
    print(len(objmeta.class_list))

    import random

    oop_metadata = MetadataCatalog.get("oop_train")
    dataset_dicts = get_oop_dicts(os.path.join(ROOT, "train"))
    for id, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=oop_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(f"./show_{id}.png", out.get_image()[:, :, ::-1])
        print(d["file_name"])
    print("done.")
