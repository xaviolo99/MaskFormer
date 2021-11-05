import io
import time
from collections import namedtuple

import numpy as np
import torch
from PIL import Image
import zstd
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

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
        self.projections = projections

    def __len__(self):
        return len(self.panoramas)

    def __getitem__(self, idx):
        if type(self.kluster) == tuple:
            self.kluster = Kluster(session=MongoClient(*self.kluster))
        segment_id, line_idx, panorama_idx, panorama_id = self.panoramas[idx]
        panorama = self.kluster.kluster["street_view"].find_one({"_id": panorama_id})
        shards = mongo_to_shards(panorama["panorama"])
        panoramator = Panoramator(shards=shards, atomic_resolution=panorama["resolution"][0]//16)
        panoramator.build_state()
        projections = [(projection_meta, panoramator.get_projection(projection_meta))
                       for projection_meta in self.projections]
        return segment_id, line_idx, panorama_id, projections


def inference(kluster, predictor, data_loader, keyword):
    current_line = None
    line_count = 0

    for i, (segment_id, line_idx, panorama_id, projections) in enumerate(data_loader):
        itime = time.time()

        if current_line is not None and current_line != (segment_id, line_idx):
            sid, lidx = current_line
            kluster.kluster["segments"].update_one({"_id": sid}, {"$set": {f"street_view.{lidx}.{keyword}": True}})
            line_count += 1
            print(f"Finished line {line_count}! (Segment:{sid};Index:{lidx})")
        current_line = (segment_id, line_idx)

        result = []
        for projection_meta, projection in projections:
            predictions = predictor(projection)
            result.append({"projection": projection_meta.get_dict(), **predictions})
        kluster.kluster["street_view"].update_one({"_id": panorama_id}, {"$set": {keyword: result}})

        print(f"Predicted panorama {i+1}/{len(data_loader)} (Time elapsed: {time.time()-itime:.2f}s) ({panorama_id})")


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
def process_image(model, image, quantization):
    # t0 = time.time()
    if type(image) == torch.Tensor:  # When using a DataLoader, Tensors instead of arrays will be given
        image = image.numpy()
    image = image[:, :, ::-1]  # VERY IMPORTANT! CONVERT IMAGE FROM RGB (PIL format) TO BGR (model format)
    predictions = model(image)
    # t1 = time.time()
    segmentation = predictions["sem_seg"].argmax(dim=0).to(torch.uint8).cpu()
    segmentation = np.moveaxis(segmentation.numpy(), 0, -1)
    buffer = io.BytesIO()
    Image.fromarray(segmentation).save(buffer, format="PNG")
    segmentation = buffer.getvalue()
    # t2 = time.time()
    probability_maps = (predictions["sem_seg"] * quantization).to(torch.uint8)
    probability_maps[:, 1:, :] = torch.diff(probability_maps, dim=1)
    probability_maps = probability_maps.cpu().numpy()
    buffer = io.BytesIO()
    np.save(buffer, probability_maps)
    probability_maps = buffer.getvalue()
    # t3 = time.time()
    probability_maps = zstd.compress(probability_maps, 9)
    # t4 = time.time()
    # print(f"Model: {(t1 - t0) * 1000:.0f} ms; Segmentation: {(t2 - t1) * 1000:.0f} ms; "
    #       f"Probability maps: {(t3 - t2) * 1000:.0f} ms; Compression {(t4 - t3) * 1000:.0f} ms;")
    return {"segmentation": segmentation, "probability_maps": probability_maps}


# Constants

Args = namedtuple("Args", "config_file opts")
CONFIG = "configs/mapillary-vistas-65-v2/maskformer_panoptic_swin_base_transfer.yaml"
WEIGHTS = "output/model_final.pth"
MONGO_SESSION_ARGS = ("localhost", 27017)
PREDICTION_KEYWORD = "mapillary_semantic"
TIMEOUT = 180
PROJECTIONS = [Projection(center_horizontal=0, center_vertical=0, fov_horizontal=92.5, fov_vertical=71.37,
                          full_resolution_x=1280, full_resolution_y=880,
                          offset_x=0, offset_y=880-640, resolution_x=1280, resolution_y=640),
               Projection(center_horizontal=180, center_vertical=0, fov_horizontal=92.5, fov_vertical=71.37,
                          full_resolution_x=1280, full_resolution_y=880,
                          offset_x=0, offset_y=880 - 640, resolution_x=1280, resolution_y=640)]
MIN_LAT, MAX_LAT = 41.35, 41.5
MIN_LON, MAX_LON = 2.1, 2.3
QUANTIZATION = 40
# CATEGORIES = [category["name"] for category in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]


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
    vistas_model = DefaultPredictor(setup_cfg(vistas_args))  # VisualizationDemo(setup_cfg(vistas_args))
    vistas_predictor = lambda image: process_image(vistas_model, image, QUANTIZATION)

    # Load segment_ids of interest
    ways = main_kluster.fetch_data("ways", {"path": {"$geoIntersects": {"$geometry": bounding_polygon}}})
    segment_ids = [seg_id for way in ways for seg_id in way["segments"].values()]

    # Do the inference, and when it finishes keep looking for new panoramas
    while True:
        dataset = PanoramaDataset(MONGO_SESSION_ARGS, segment_ids, PREDICTION_KEYWORD, PROJECTIONS)
        if len(dataset):
            print(f"LAUNCHING INFERENCE ON {len(dataset)} PANORAMAS")
            loader = DataLoader(dataset, batch_size=None, num_workers=4)
            inference(main_kluster, vistas_predictor, loader, PREDICTION_KEYWORD)
        else:
            print(f"NO PANORAMAS FOUND! WAITING {TIMEOUT} seconds...")
            time.sleep(180)
