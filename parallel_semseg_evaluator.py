# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator

from pathos.multiprocessing import Pool


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        processes=4
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        self.pool = Pool(processes)

    def reset(self):
        self._tasks = []
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        if len(self._tasks) >= 100:
            while len(self._tasks):
                task = self._tasks.pop(0)
                conf_matrix, predictions = task.get()
                self._conf_matrix += conf_matrix
                self._predictions.extend(predictions)
        outputs = [output["sem_seg"].argmax(dim=0).to(self._cpu_device) for output in outputs]
        task = self.pool.apply_async(_process, (inputs, outputs, self._num_classes, self._ignore_label,
                                                self.input_file_to_gt_file, self._contiguous_id_to_dataset_id))
        self._tasks.append(task)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        for task in self._tasks:
            conf_matrix, predictions = task.get()
            self._conf_matrix += conf_matrix
            self._predictions.extend(predictions)

        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results


def _process(inputs, outputs, _num_classes, _ignore_label, input_file_to_gt_file, _contiguous_id_to_dataset_id):
    """
    Args:
        inputs: the inputs to a model.
            It is a list of dicts. Each dict corresponds to an image and
            contains keys like "height", "width", "file_name".
        outputs: the outputs of a model. It is either list of semantic segmentation predictions
            (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
            segmentation prediction in the same format.
    """
    conf_matrix = np.zeros((_num_classes + 1, _num_classes + 1), dtype=np.int64)
    predictions = []

    for input, output in zip(inputs, outputs):
        # output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
        pred = np.array(output, dtype=np.int)
        with PathManager.open(input_file_to_gt_file[input["file_name"]], "rb") as f:
            gt = np.array(Image.open(f), dtype=np.int)

        gt[gt == _ignore_label] = _num_classes

        conf_matrix += np.bincount(
            (_num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix.size,
        ).reshape(conf_matrix.shape)

        predictions.extend(encode_json_sem_seg(pred, input["file_name"], _contiguous_id_to_dataset_id))

    return conf_matrix, predictions


def encode_json_sem_seg(sem_seg, input_file_name, _contiguous_id_to_dataset_id):
    """
    Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
    See http://cocodataset.org/#format-results
    """
    json_list = []
    for label in np.unique(sem_seg):
        if _contiguous_id_to_dataset_id is not None:
            assert (
                label in _contiguous_id_to_dataset_id
            ), "Label {} is not in the metadata info".format(label)
            dataset_id = _contiguous_id_to_dataset_id[label]
        else:
            dataset_id = int(label)
        mask = (sem_seg == label).astype(np.uint8)
        mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
        mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
        json_list.append(
            {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
        )
    return json_list
