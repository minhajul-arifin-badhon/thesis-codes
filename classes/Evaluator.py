# Evaluator combines the ability to both predict and calculate tp, fp, precision, recall and ap.
# If det boxes are already calculated we can use methods from utils.evaluation and pass gt boxes and det boxes
import mmcv
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import DOTA_devkit.polyiou as polyiou
from IPython.display import display, HTML

import sys
sys.path.append('/home/badhon/Documents/thesis/thesis-codes/')
from helpers import utils
from helpers import evaluation
from classes.WheatDataset import WheatDataset

sys.path.append("/home/badhon/Documents/thesis/AerialDetection")
import DOTA_devkit.polyiou as polyiou
from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections


class Evaluator:
    def __init__(self, path_config, path_work_dir, epoch):
        self._path_config = path_config
        self._path_work_dir = path_work_dir
        self.build_detector_from_epoch(epoch)

    def build_detector_from_epoch(self, epoch):
        self._epoch = epoch
        self._model =  init_detector(self._path_config, os.path.join(self._path_work_dir, 'epoch_' + str(epoch) + '.pth'), device='cuda:0')

    def predict_single_image(self, path_image, display_predicition=False, r_dict=False):
        detections = np.array(inference_detector(self._model, path_image)[0])
        if detections.shape[1] <= 5:
#             print(detections)
            confidence = np.array([detections[:,4]])
            bboxes = utils.xyxy_to_poly(detections[:,:4], True)
            detections = np.concatenate((bboxes, confidence.T), axis=1)
#             print(detections)

        if display_predicition:
            utils.show_poly_anno(
                path_image, [detections], ["Detections: " + path_image]
            )

        if r_dict:
            detections = {
                path_image: detections
            }

        return detections

    def file_range(self, path_folder, limit):
        filenames = os.listdir(path_folder)
        start = limit[0] if limit else 0
        end = limit[1] if limit else len(filenames)
        return filenames[start:end]

    def predict_image_folder(self, path_folder, limit=None, display_predicition=False):
        filenames = self.file_range(path_folder, limit)

        all_dets = {}
        for path_image in filenames:
            all_dets[path_image] = self.predict_single_image(os.path.join(path_folder, path_image), display_predicition)
        return all_dets

    def evaluate_single_image(self, path_image, gt_boxes, thresh=0.5, use_07_metric=False, r_metrics = None):
        all_gts = {
            path_image: gt_boxes
        }
        all_dets = self.predict_single_image(path_image, False, True)
        res = evaluation.voc_eval(all_gts, all_dets, thresh, use_07_metric)

        if r_metrics:
            res = [res['metric'] for metric in r_metrics]
        return res

    def evaluate_image_folder(self, path_folder, limit=None, all_gts={}, thresh=0.5, use_07_metric=False, r_metrics = None):
        filenames = self.file_range(path_folder, limit)

        for key in list(all_gts.keys()):
            if key not in filenames:
                all_gts.pop(key, None)

        all_dets = self.predict_image_folder(path_folder, limit)

        res = evaluation.voc_eval(all_gts, all_dets, thresh, use_07_metric)

        if r_metrics:
            res = [res['metric'] for metric in r_metrics]
        return res

    def print_summary(self, res={}, exclude=['rec', 'prec']):
        for key in res.keys():
            if key in exclude:
                continue
            print(key + ": ", res[key])
