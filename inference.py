import io
import time
from collections import namedtuple

import numpy as np
import torch
from PIL import Image
import zstd
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from kluster import Kluster
from panoramator import Projection, Panoramator, mongo_to_shards

from mask_former import add_mask_former_config
from mask_former.data.datasets.register_mapillary_vistas import MAPILLARY_VISTAS_SEM_SEG_CATEGORIES
from demo.predictor import VisualizationDemo


# Panoramator structures

class PanoramaDataset(Dataset):

    def __init__(self, mongo_args, segments, keyword, projections):
        kluster = Kluster(session=MongoClient(*mongo_args))
        segments = kluster.fetch_data(
            "segments",
            {"_id": {"$in": segments}, "street_view": {"$elemMatch": {"available": True, keyword: {"$exists": False}}}}
        )
        self.kluster = mongo_args
        lines = [
            (segment["_id"], i, line["panoramas"])
            for segment in segments for i, line in enumerate(segment["street_view"])
            if "available" in line and keyword not in line
        ]
        self.panoramas = [(sid, lidx, pidx, panorama)
                          for sid, lidx, panoramas in lines for pidx, panorama in enumerate(panoramas)]
        self.keyword = keyword
        self.projections = projections

    def __len__(self):
        return len(self.panoramas)

    def __getitem__(self, idx):
        if type(self.kluster) == tuple:
            self.kluster = Kluster(session=MongoClient(*self.kluster))
        segment_id, line_idx, panorama_idx, panorama_id = self.panoramas[idx]
        panorama = self.kluster.kluster["street_view"].find_one({"_id": panorama_id})
        if self.keyword in panorama:  # Escape if this panorama is already predicted (but line was not marked)
            return segment_id, line_idx, panorama_id, None
        shards = mongo_to_shards(panorama["image_shards"])
        panoramator = Panoramator(shards=shards, atomic_resolution=panorama["resolution"][0] // 16)
        panoramator.build_state()
        projections = [(projection_meta, panoramator.get_projection(projection_meta))
                       for projection_meta in self.projections]
        return segment_id, line_idx, panorama_id, projections


def inference(kluster, predictor, data_loader, keyword, upload=True):
    current_line = None
    line_count = 0
    itime = time.time()

    for i, (segment_id, line_idx, panorama_id, projections) in enumerate(data_loader):

        if current_line is not None and current_line != (segment_id, line_idx):
            sid, lidx = current_line
            if upload:
                kluster.kluster["segments"].update_one({"_id": sid}, {"$set": {f"street_view.{lidx}.{keyword}": True}})
            line_count += 1
            print(f"Finished line {line_count}! (Segment:{sid};Index:{lidx})")
        current_line = (segment_id, line_idx)

        if projections is not None:  # If the panorama is already predicted, we skip this block
            result = []
            for projection_meta, projection in projections:
                predictions = predictor(projection)
                result.append({"projection": projection_meta.get_dict(), **predictions})
            if upload:
                kluster.kluster["street_view"].update_one({"_id": panorama_id}, {"$set": {keyword: result}})

        print(f"Predicted panorama {i + 1}/{len(data_loader)} "
              f"(Time elapsed: {time.time() - itime:.2f}s) ({panorama_id})")
        itime = time.time()

    if current_line is not None:
        sid, lidx = current_line
        if upload:
            kluster.kluster["segments"].update_one({"_id": sid}, {"$set": {f"street_view.{lidx}.{keyword}": True}})
        line_count += 1
        print(f"Finished line {line_count}! (Segment:{sid};Index:{lidx})")


# MaskFormer structures

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


# TO PARALLELIZE:
# THE OBJECTIVE IS TO MOVE THE KLUSTER UPDATING FUNCTION TO PROCESS IMAGE (PUSH RESULT TO ARRAY)
# ONCE WE HAVE THIS, WE START A PROCESS WHEN THE GPU ENDS INFERENCE THAT WILL DO THE POSTPROCESSING
# WE RETURN THIS PROCESS TO THE MAIN INFERENCE LOOP, AND WE KEEP A QUEUE OF PROCESSES THERE THAT WE UPDATE
# THIS QUEUE WILL HAVE A FIXED SIZE (FOR INSTANCE 4), AND WHEN WE ADD THE NEW PROCESS, WE WILL JOIN AND DELETE THE FIRST
# IN https://docs.python.org/3/library/multiprocessing.html UNDER `The Process` class IS ALL THE INFO NEEDED
def process_image(model, image, plot=False):
    t0 = time.time()

    if type(image) == torch.Tensor:  # When using a DataLoader, Tensors instead of arrays will be given
        image = image.numpy()
    image = image[:, :, ::-1]  # VERY IMPORTANT! CONVERT IMAGE FROM RGB (PIL format) TO BGR (model format)
    predictions = model(image)
    t1 = time.time()

    predictions["sem_seg"][36, :, :] *= 0  # object--catch-basin
    predictions["sem_seg"][41, :, :] *= 0  # object--manhole
    predictions["sem_seg"][43, :, :] *= 0  # object--pothole
    sem_seg_parking = predictions["sem_seg"]
    sem_seg_marking = torch.clone(sem_seg_parking)
    sem_seg_parking[10, :, :] *= 2  # construction--flat--parking
    sem_seg_parking[24, :, :] *= 0  # marking--general
    seg_parking = sem_seg_parking.argmax(dim=0).to(torch.uint8)
    sem_seg_marking[24, :, :] *= 1  # marking--general
    seg_marking = sem_seg_marking.argmax(dim=0).to(torch.uint8)
    seg = torch.where(torch.logical_and(seg_parking == 10, seg_marking == 24),
                      torch.ByteTensor([66])[0].cuda(), seg_parking)
    seg = torch.where(torch.logical_and(seg_parking != 10, seg_marking == 24),
                      torch.ByteTensor([24])[0].cuda(), seg)
    t2 = time.time()

    segmentation = np.moveaxis(seg.cpu().numpy(), 0, -1)
    buffer = io.BytesIO()
    Image.fromarray(segmentation).save(buffer, format="PNG")
    segmentation = buffer.getvalue()
    t3 = time.time()

    # print(f"Model: {(t1 - t0) * 1000:.0f} ms; Segmentation: {(t2 - t1) * 1000:.0f} ms; "
    #       f"Saving: {(t3 - t2) * 1000:.0f} ms;  Image size {len(segmentation)/1e3:.0f} KB;")

    if plot:
        plt.rcParams["figure.figsize"] = [10, 5]
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        visualizer = Visualizer(image[:, :, ::-1], metadata)
        seg = torch.where(seg == 66, torch.ByteTensor([64])[0].cuda(), seg)
        vis_output = visualizer.draw_sem_seg(seg.cpu())
        plt.imshow(vis_output.get_image())
        plt.show()

    return {"segmentation": segmentation}
    # return {"probability_maps": probability_maps}


CATEGORY_MAP = {
    'animal--bird': 0,
    'animal--ground-animal': 1,
    'construction--barrier--curb': 2,
    'construction--barrier--fence': 3,
    'construction--barrier--guard-rail': 4,
    'construction--barrier--other-barrier': 5,
    'construction--barrier--wall': 6,
    'construction--flat--bike-lane': 7,
    'construction--flat--crosswalk-plain': 8,
    'construction--flat--curb-cut': 9,
    'construction--flat--parking': 10,
    'construction--flat--pedestrian-area': 11,
    'construction--flat--rail-track': 12,
    'construction--flat--road': 13,
    'construction--flat--service-lane': 14,
    'construction--flat--sidewalk': 15,
    'construction--structure--bridge': 16,
    'construction--structure--building': 17,
    'construction--structure--tunnel': 18,
    'human--person': 19,
    'human--rider--bicyclist': 20,
    'human--rider--motorcyclist': 21,
    'human--rider--other-rider': 22,
    'marking--crosswalk-zebra': 23,
    'marking--general': 24,
    'nature--mountain': 25,
    'nature--sand': 26,
    'nature--sky': 27,
    'nature--snow': 28,
    'nature--terrain': 29,
    'nature--vegetation': 30,
    'nature--water': 31,
    'object--banner': 32,
    'object--bench': 33,
    'object--bike-rack': 34,
    'object--billboard': 35,
    'object--catch-basin': 36,
    'object--cctv-camera': 37,
    'object--fire-hydrant': 38,
    'object--junction-box': 39,
    'object--mailbox': 40,
    'object--manhole': 41,
    'object--phone-booth': 42,
    'object--pothole': 43,
    'object--street-light': 44,
    'object--support--pole': 45,
    'object--support--traffic-sign-frame': 46,
    'object--support--utility-pole': 47,
    'object--traffic-light': 48,
    'object--traffic-sign--back': 49,
    'object--traffic-sign--front': 50,
    'object--trash-can': 51,
    'object--vehicle--bicycle': 52,
    'object--vehicle--boat': 53,
    'object--vehicle--bus': 54,
    'object--vehicle--car': 55,
    'object--vehicle--caravan': 56,
    'object--vehicle--motorcycle': 57,
    'object--vehicle--on-rails': 58,
    'object--vehicle--other-vehicle': 59,
    'object--vehicle--trailer': 60,
    'object--vehicle--truck': 61,
    'object--vehicle--wheeled-slow': 62,
    'void--car-mount': 63,
    'void--ego-vehicle': 64,
    'void--unlabeled': 65,
    'marking--parking': 66,  # THIS LAST ONE IS ARTIFICIALLY ADDED
}

# Constants

Args = namedtuple("Args", "config_file opts")
CONFIG = "configs/mapillary-vistas-65-v2/maskformer_panoptic_swin_base_transfer.yaml"
WEIGHTS = "output/model_final.pth"
MONGO_SESSION_ARGS = ("localhost", 27017)
PREDICTION_KEYWORD = "mapillary_semantic"
TIMEOUT = 180
PROJECTIONS = [Projection(center_horizontal=0, center_vertical=0, fov_horizontal=92.5, fov_vertical=71.37,
                          full_resolution_x=1280, full_resolution_y=880,
                          offset_x=0, offset_y=880 - 640, resolution_x=1280, resolution_y=640),
               Projection(center_horizontal=180, center_vertical=0, fov_horizontal=92.5, fov_vertical=71.37,
                          full_resolution_x=1280, full_resolution_y=880,
                          offset_x=0, offset_y=880 - 640, resolution_x=1280, resolution_y=640)]
MIN_LAT, MAX_LAT = 41.35, 41.5
MIN_LON, MAX_LON = 2.1, 2.3
PLOT = False
UPLOAD = True

# Main Execution

if __name__ == "__main__":
    # StreetView initializations
    main_kluster = Kluster(session=MongoClient(*MONGO_SESSION_ARGS))
    bounding_polygon = [(MIN_LAT, MIN_LON), (MIN_LAT, MAX_LON), (MAX_LAT, MAX_LON),
                        (MAX_LAT, MIN_LON), (MIN_LAT, MIN_LON)]
    bounding_polygon = {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in bounding_polygon]]}

    # MaskFormer initializations
    vistas_args = Args(CONFIG, opts=["MODEL.WEIGHTS", WEIGHTS])
    setup_logger(name="fvcore")
    logger = setup_logger()
    cfg = setup_cfg(vistas_args)
    vistas_model = DefaultPredictor(cfg)  # VisualizationDemo(setup_cfg(vistas_args))
    vistas_predictor = lambda image: process_image(vistas_model, image, plot=PLOT)

    # Load segment_ids of interest
    ways = main_kluster.fetch_data("ways", {"path": {"$geoIntersects": {"$geometry": bounding_polygon}}})
    segment_ids = [seg_id for way in ways for seg_id in way["segments"].values()]

    # Do the inference, and when it finishes keep looking for new panoramas
    while True:
        dataset = PanoramaDataset(MONGO_SESSION_ARGS, segment_ids, PREDICTION_KEYWORD, PROJECTIONS)
        if len(dataset):
            print(f"LAUNCHING INFERENCE ON {len(dataset)} PANORAMAS")
            loader = DataLoader(dataset, batch_size=None, num_workers=4)
            inference(main_kluster, vistas_predictor, loader, PREDICTION_KEYWORD, upload=UPLOAD)
        else:
            print(f"NO PANORAMAS FOUND! WAITING {TIMEOUT} seconds...")
            time.sleep(180)
