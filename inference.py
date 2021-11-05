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
def process_image(model, image, quantization, plot=False):
    if type(image) == torch.Tensor:  # When using a DataLoader, Tensors instead of arrays will be given
        image = image.numpy()
    image = image[:, :, ::-1]  # VERY IMPORTANT! CONVERT IMAGE FROM RGB (PIL format) TO BGR (model format)
    predictions = model(image)

    # segmentation = predictions["sem_seg"].argmax(dim=0).to(torch.uint8).cpu()
    # segmentation = np.moveaxis(segmentation.numpy(), 0, -1)
    # buffer = io.BytesIO()
    # Image.fromarray(segmentation).save(buffer, format="PNG")
    # segmentation = buffer.getvalue()

    probability_maps = (predictions["sem_seg"] * quantization).to(torch.uint8)
    probability_maps[:, 1:, :] = torch.diff(probability_maps, dim=1)
    probability_maps = probability_maps.cpu().numpy()
    buffer = io.BytesIO()
    np.save(buffer, probability_maps)
    probability_maps = buffer.getvalue()

    probability_maps = zstd.compress(probability_maps, 9)

    # print(f"Model: {(t1 - t0) * 1000:.0f} ms; Segmentation: {(t2 - t1) * 1000:.0f} ms; "
    #       f"Probability maps: {(t3 - t2) * 1000:.0f} ms; Compression {(t4 - t3) * 1000:.0f} ms;")
    if plot:
        predictions["sem_seg"][10, :, :] *= 4  # parking
        plt.rcParams['figure.figsize'] = [10, 5]
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        visualizer = Visualizer(image[:, :, ::-1], metadata)
        vis_output = visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).cpu())
        plt.imshow(vis_output.get_image())
        plt.show()

    # return {"segmentation": segmentation, "probability_maps": probability_maps}
    return {"probability_maps": probability_maps}


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
PLOT = False


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
    vistas_predictor = lambda image: process_image(vistas_model, image, QUANTIZATION, plot=PLOT)

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

"""
[(0,
  {'color': [165, 42, 42],
   'instances': True,
   'readable': 'Bird',
   'name': 'animal--bird',
   'evaluate': True}),
 (1,
  {'color': [0, 192, 0],
   'instances': True,
   'readable': 'Ground Animal',
   'name': 'animal--ground-animal',
   'evaluate': True}),
 (2,
  {'color': [196, 196, 196],
   'instances': False,
   'readable': 'Curb',
   'name': 'construction--barrier--curb',
   'evaluate': True}),
 (3,
  {'color': [190, 153, 153],
   'instances': False,
   'readable': 'Fence',
   'name': 'construction--barrier--fence',
   'evaluate': True}),
 (4,
  {'color': [180, 165, 180],
   'instances': False,
   'readable': 'Guard Rail',
   'name': 'construction--barrier--guard-rail',
   'evaluate': True}),
 (5,
  {'color': [90, 120, 150],
   'instances': False,
   'readable': 'Barrier',
   'name': 'construction--barrier--other-barrier',
   'evaluate': True}),
 (6,
  {'color': [102, 102, 156],
   'instances': False,
   'readable': 'Wall',
   'name': 'construction--barrier--wall',
   'evaluate': True}),
 (7,
  {'color': [128, 64, 255],
   'instances': False,
   'readable': 'Bike Lane',
   'name': 'construction--flat--bike-lane',
   'evaluate': True}),
 (8,
  {'color': [140, 140, 200],
   'instances': True,
   'readable': 'Crosswalk - Plain',
   'name': 'construction--flat--crosswalk-plain',
   'evaluate': True}),
 (9,
  {'color': [170, 170, 170],
   'instances': False,
   'readable': 'Curb Cut',
   'name': 'construction--flat--curb-cut',
   'evaluate': True}),
 (10,
  {'color': [250, 170, 160],
   'instances': False,
   'readable': 'Parking',
   'name': 'construction--flat--parking',
   'evaluate': True}),
 (11,
  {'color': [96, 96, 96],
   'instances': False,
   'readable': 'Pedestrian Area',
   'name': 'construction--flat--pedestrian-area',
   'evaluate': True}),
 (12,
  {'color': [230, 150, 140],
   'instances': False,
   'readable': 'Rail Track',
   'name': 'construction--flat--rail-track',
   'evaluate': True}),
 (13,
  {'color': [128, 64, 128],
   'instances': False,
   'readable': 'Road',
   'name': 'construction--flat--road',
   'evaluate': True}),
 (14,
  {'color': [110, 110, 110],
   'instances': False,
   'readable': 'Service Lane',
   'name': 'construction--flat--service-lane',
   'evaluate': True}),
 (15,
  {'color': [244, 35, 232],
   'instances': False,
   'readable': 'Sidewalk',
   'name': 'construction--flat--sidewalk',
   'evaluate': True}),
 (16,
  {'color': [150, 100, 100],
   'instances': False,
   'readable': 'Bridge',
   'name': 'construction--structure--bridge',
   'evaluate': True}),
 (17,
  {'color': [70, 70, 70],
   'instances': False,
   'readable': 'Building',
   'name': 'construction--structure--building',
   'evaluate': True}),
 (18,
  {'color': [150, 120, 90],
   'instances': False,
   'readable': 'Tunnel',
   'name': 'construction--structure--tunnel',
   'evaluate': True}),
 (19,
  {'color': [220, 20, 60],
   'instances': True,
   'readable': 'Person',
   'name': 'human--person',
   'evaluate': True}),
 (20,
  {'color': [255, 0, 0],
   'instances': True,
   'readable': 'Bicyclist',
   'name': 'human--rider--bicyclist',
   'evaluate': True}),
 (21,
  {'color': [255, 0, 100],
   'instances': True,
   'readable': 'Motorcyclist',
   'name': 'human--rider--motorcyclist',
   'evaluate': True}),
 (22,
  {'color': [255, 0, 200],
   'instances': True,
   'readable': 'Other Rider',
   'name': 'human--rider--other-rider',
   'evaluate': True}),
 (23,
  {'color': [200, 128, 128],
   'instances': True,
   'readable': 'Lane Marking - Crosswalk',
   'name': 'marking--crosswalk-zebra',
   'evaluate': True}),
 (24,
  {'color': [255, 255, 255],
   'instances': False,
   'readable': 'Lane Marking - General',
   'name': 'marking--general',
   'evaluate': True}),
 (25,
  {'color': [64, 170, 64],
   'instances': False,
   'readable': 'Mountain',
   'name': 'nature--mountain',
   'evaluate': True}),
 (26,
  {'color': [230, 160, 50],
   'instances': False,
   'readable': 'Sand',
   'name': 'nature--sand',
   'evaluate': True}),
 (27,
  {'color': [70, 130, 180],
   'instances': False,
   'readable': 'Sky',
   'name': 'nature--sky',
   'evaluate': True}),
 (28,
  {'color': [190, 255, 255],
   'instances': False,
   'readable': 'Snow',
   'name': 'nature--snow',
   'evaluate': True}),
 (29,
  {'color': [152, 251, 152],
   'instances': False,
   'readable': 'Terrain',
   'name': 'nature--terrain',
   'evaluate': True}),
 (30,
  {'color': [107, 142, 35],
   'instances': False,
   'readable': 'Vegetation',
   'name': 'nature--vegetation',
   'evaluate': True}),
 (31,
  {'color': [0, 170, 30],
   'instances': False,
   'readable': 'Water',
   'name': 'nature--water',
   'evaluate': True}),
 (32,
  {'color': [255, 255, 128],
   'instances': True,
   'readable': 'Banner',
   'name': 'object--banner',
   'evaluate': True}),
 (33,
  {'color': [250, 0, 30],
   'instances': True,
   'readable': 'Bench',
   'name': 'object--bench',
   'evaluate': True}),
 (34,
  {'color': [100, 140, 180],
   'instances': True,
   'readable': 'Bike Rack',
   'name': 'object--bike-rack',
   'evaluate': True}),
 (35,
  {'color': [220, 220, 220],
   'instances': True,
   'readable': 'Billboard',
   'name': 'object--billboard',
   'evaluate': True}),
 (36,
  {'color': [220, 128, 128],
   'instances': True,
   'readable': 'Catch Basin',
   'name': 'object--catch-basin',
   'evaluate': True}),
 (37,
  {'color': [222, 40, 40],
   'instances': True,
   'readable': 'CCTV Camera',
   'name': 'object--cctv-camera',
   'evaluate': True}),
 (38,
  {'color': [100, 170, 30],
   'instances': True,
   'readable': 'Fire Hydrant',
   'name': 'object--fire-hydrant',
   'evaluate': True}),
 (39,
  {'color': [40, 40, 40],
   'instances': True,
   'readable': 'Junction Box',
   'name': 'object--junction-box',
   'evaluate': True}),
 (40,
  {'color': [33, 33, 33],
   'instances': True,
   'readable': 'Mailbox',
   'name': 'object--mailbox',
   'evaluate': True}),
 (41,
  {'color': [100, 128, 160],
   'instances': True,
   'readable': 'Manhole',
   'name': 'object--manhole',
   'evaluate': True}),
 (42,
  {'color': [142, 0, 0],
   'instances': True,
   'readable': 'Phone Booth',
   'name': 'object--phone-booth',
   'evaluate': True}),
 (43,
  {'color': [70, 100, 150],
   'instances': False,
   'readable': 'Pothole',
   'name': 'object--pothole',
   'evaluate': True}),
 (44,
  {'color': [210, 170, 100],
   'instances': True,
   'readable': 'Street Light',
   'name': 'object--street-light',
   'evaluate': True}),
 (45,
  {'color': [153, 153, 153],
   'instances': True,
   'readable': 'Pole',
   'name': 'object--support--pole',
   'evaluate': True}),
 (46,
  {'color': [128, 128, 128],
   'instances': True,
   'readable': 'Traffic Sign Frame',
   'name': 'object--support--traffic-sign-frame',
   'evaluate': True}),
 (47,
  {'color': [0, 0, 80],
   'instances': True,
   'readable': 'Utility Pole',
   'name': 'object--support--utility-pole',
   'evaluate': True}),
 (48,
  {'color': [250, 170, 30],
   'instances': True,
   'readable': 'Traffic Light',
   'name': 'object--traffic-light',
   'evaluate': True}),
 (49,
  {'color': [192, 192, 192],
   'instances': True,
   'readable': 'Traffic Sign (Back)',
   'name': 'object--traffic-sign--back',
   'evaluate': True}),
 (50,
  {'color': [220, 220, 0],
   'instances': True,
   'readable': 'Traffic Sign (Front)',
   'name': 'object--traffic-sign--front',
   'evaluate': True}),
 (51,
  {'color': [140, 140, 20],
   'instances': True,
   'readable': 'Trash Can',
   'name': 'object--trash-can',
   'evaluate': True}),
 (52,
  {'color': [119, 11, 32],
   'instances': True,
   'readable': 'Bicycle',
   'name': 'object--vehicle--bicycle',
   'evaluate': True}),
 (53,
  {'color': [150, 0, 255],
   'instances': True,
   'readable': 'Boat',
   'name': 'object--vehicle--boat',
   'evaluate': True}),
 (54,
  {'color': [0, 60, 100],
   'instances': True,
   'readable': 'Bus',
   'name': 'object--vehicle--bus',
   'evaluate': True}),
 (55,
  {'color': [0, 0, 142],
   'instances': True,
   'readable': 'Car',
   'name': 'object--vehicle--car',
   'evaluate': True}),
 (56,
  {'color': [0, 0, 90],
   'instances': True,
   'readable': 'Caravan',
   'name': 'object--vehicle--caravan',
   'evaluate': True}),
 (57,
  {'color': [0, 0, 230],
   'instances': True,
   'readable': 'Motorcycle',
   'name': 'object--vehicle--motorcycle',
   'evaluate': True}),
 (58,
  {'color': [0, 80, 100],
   'instances': False,
   'readable': 'On Rails',
   'name': 'object--vehicle--on-rails',
   'evaluate': True}),
 (59,
  {'color': [128, 64, 64],
   'instances': True,
   'readable': 'Other Vehicle',
   'name': 'object--vehicle--other-vehicle',
   'evaluate': True}),
 (60,
  {'color': [0, 0, 110],
   'instances': True,
   'readable': 'Trailer',
   'name': 'object--vehicle--trailer',
   'evaluate': True}),
 (61,
  {'color': [0, 0, 70],
   'instances': True,
   'readable': 'Truck',
   'name': 'object--vehicle--truck',
   'evaluate': True}),
 (62,
  {'color': [0, 0, 192],
   'instances': True,
   'readable': 'Wheeled Slow',
   'name': 'object--vehicle--wheeled-slow',
   'evaluate': True}),
 (63,
  {'color': [32, 32, 32],
   'instances': False,
   'readable': 'Car Mount',
   'name': 'void--car-mount',
   'evaluate': True}),
 (64,
  {'color': [120, 10, 10],
   'instances': False,
   'readable': 'Ego Vehicle',
   'name': 'void--ego-vehicle',
   'evaluate': True}),
 (65,
  {'color': [0, 0, 0],
   'instances': False,
   'readable': 'Unlabeled',
   'name': 'void--unlabeled',
   'evaluate': False})]
"""