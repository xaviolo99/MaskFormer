import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

###
# Load metadata and set constants
###

# ROOT = "."
ROOT = os.getenv("DETECTRON2_DATASETS", "datasets")
DATASET = "detectron_vistas"

with open(os.path.join(ROOT, DATASET, "config.json"), "r") as handler:
    labels_meta = json.load(handler)["labels"]
labels_meta = [{"name": "unlabeled", "readable": "unlabeled", "instances": False,
                "evaluate": True, "color": [0, 0, 0]}] + labels_meta
labels_meta = {label["name"]: label for label in labels_meta}

ORDERED_LABELS = [label["name"] for label in labels_meta.values() if label["evaluate"]]

THING = [label["name"] for label in labels_meta.values()
         if label["instances"] and label["evaluate"]]
STUFF = [label["name"] for label in labels_meta.values()
         if not label["instances"] and label["evaluate"]]

labels_meta_id = {i: labels_meta[name] for i, name in enumerate(ORDERED_LABELS)}


###
# Dataset data
###

def create_mapillary_dataset(image_dir, annotation_dir, json_dir):
    with open(json_dir, "r") as handler:
        metadata = json.load(handler)
    annotations = metadata["annotations"]
    images = {image["id"]: image for image in metadata["images"]}

    mapillary_dataset = []
    for annotation in annotations:
        image_id = annotation["image_id"]
        height = images[image_id]["height"]
        width = images[image_id]["width"]
        image_file = os.path.join(image_dir, images[image_id]["file_name"])
        annotation_file = os.path.join(annotation_dir, annotation["file_name"])
        segments_info = [{"id": ann["id"], "category_id": ann["category_id"],
                          "iscrowd": ann["iscrowd"], "isthing": labels_meta_id[ann["category_id"]]["instances"]}
                         for ann in annotation["segments_info"]]
        mapillary_dataset.append({
            "image_id": image_id,
            "height": height,
            "width": width,
            "file_name": image_file,
            "pan_seg_file_name": annotation_file,
            "segments_info": segments_info,
        })

    return mapillary_dataset


###
# Dataset metadata
###

thing_classes = [label for label in ORDERED_LABELS if label in THING + STUFF]
thing_colors = [tuple(labels_meta[label]["color"]) for label in thing_classes]
stuff_classes = [label for label in ORDERED_LABELS if label in THING + STUFF]
stuff_colors = [tuple(labels_meta[label]["color"]) for label in stuff_classes]

thing_dataset_id_to_contiguous_id = {}
stuff_dataset_id_to_contiguous_id = {}
for i, label_name in enumerate(ORDERED_LABELS):
    if labels_meta[label_name]["instances"] and label_name in thing_classes:
        thing_dataset_id_to_contiguous_id[i] = i
    elif label_name in stuff_classes:
        stuff_dataset_id_to_contiguous_id[i] = i

###
# Creation
###

SPLITS = {
    "vistas_train": (
        os.path.join(ROOT, DATASET, "training/images"),
        os.path.join(ROOT, DATASET, "training/panoptic"),
        os.path.join(ROOT, DATASET, "training/panoptic.json"),
    ),
    "vistas_val": (
        os.path.join(ROOT, DATASET, "validation/images"),
        os.path.join(ROOT, DATASET, "validation/panoptic"),
        os.path.join(ROOT, DATASET, "validation/panoptic.json"),
    ),
}

# if __name__ == "__main__":
# DatasetCatalog.remove("vistas_train")
# MetadataCatalog.remove("vistas_train")
# DatasetCatalog.remove("vistas_val")
# MetadataCatalog.remove("vistas_val")

#print(len(thing_classes), thing_classes)
#print(len(stuff_classes), stuff_classes)
#print(len(stuff_colors), stuff_colors)

#print(len(thing_dataset_id_to_contiguous_id))
#print(len(stuff_dataset_id_to_contiguous_id))
#raise Exception()

for name, (image_dir, annotation_dir, json_dir) in SPLITS.items():
    DatasetCatalog.register(
        name,
        lambda: create_mapillary_dataset(image_dir, annotation_dir, json_dir),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        thing_colors=thing_colors,
        stuff_classes=stuff_classes,
        stuff_colors=stuff_colors,
        ignore_label=0,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        image_root=image_dir,
        panoptic_root=annotation_dir,
        panoptic_json=json_dir,
        evaluator_type="coco_panoptic_seg",
    )
