import os
import pandas as pd
import numpy as np
import json
import mmcv
import itertools
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import defaultdict
from PIL import Image

import sys
sys.path.append("/home/badhon/Documents/thesis/AerialDetection")
import DOTA_devkit.polyiou as polyiou



def xywh_to_xyxy(boxes, r_numpy=False):
    boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return boxes if r_numpy else boxes.tolist()


def xyxy_to_poly(boxes, r_numpy=False):
    boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
    poly_boxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in boxes]

    return np.array(poly_boxes) if r_numpy else poly_boxes


def get_poly_iou(box1, box2):
    box1 = np.array(box1).astype(float).tolist()
    box2 = np.array(box2).astype(float).tolist()

    return polyiou.iou_poly(polyiou.VectorDouble(box1), polyiou.VectorDouble(box2))


def show_poly_anno(filename, b_boxes=[], title=None, sep=False, f_size=(10, 10)):
    im = np.array(Image.open(filename), dtype=np.uint8)
    ln = len(b_boxes)
    fig, ax = plt.subplots(1, ln if sep else 1, figsize=f_size)

    if sep:
        for i in range(ln):
            ax[i].imshow(im)
            ax[i].set_title(title[i] if title else "")
    else:
        ax.imshow(im)
        ax.set_title(title[0] if title else "")

    color = ["w", "r"]

    for b_no, boxes in enumerate(b_boxes):
        boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
        for box in boxes:
            box_reshaped = box[:8].reshape((-1, 2))
            poly = patches.Polygon(
                box_reshaped, linewidth=1, edgecolor=color[b_no], facecolor="none",
            )

            if sep:
                ax[b_no].add_patch(poly)
            else:
                ax.add_patch(poly)

    plt.show()


# gt_boxes numpy, pred_boxes numpy
def match_detections(gt_boxes, pred_boxes, iou_thresh, show_plot = False, filename=""):
    matched = {}
    not_matched = {}
    true_positive = {}
    used = {}

    for g_idx, gt_box in enumerate(gt_boxes):
#         print("Trying to match: ")
#         print(g_idx, gt_box)
        mx_iou = 0
        idx = -1
        for p_idx, pred_box in enumerate(pred_boxes):
            if p_idx in used.keys():
                continue

            iou = get_poly_iou(gt_box, pred_box[:-1])
            if iou > iou_thresh:

                if p_idx in true_positive.keys():
                    true_positive[p_idx].append({'indx': p_idx, 'pred_box': pred_box, 'gt_box': gt_box})
                else:
                    true_positive[p_idx] = [{'indx': p_idx, 'pred_box': pred_box, 'gt_box': gt_box}]

                if iou > mx_iou:
                    matched[g_idx] = {'indx': p_idx, 'pred_box': pred_box, 'gt_box': gt_box}

                    mx_iou = iou
                    idx = p_idx

            used[idx] = True

        if idx == -1:
#             print("No Match")
            not_matched[g_idx] = {'indx': g_idx, 'gt_box': gt_box}
        else:
#             print("Found Match")
#             print(matched[g_idx])
            a = 0

    t_p = len(matched)
    f_p = pred_boxes.shape[0] - len(true_positive.keys())
    f_n = len(not_matched)
    ignored = len(true_positive.keys()) - len(matched)

    print("-------------------------------")
    print("-------------------------------")
    print("Total Ground Truth Boxes: ", gt_boxes.shape[0])
    print("Total Prediction Boxes: ", pred_boxes.shape[0])
    print("")

    print("Detections - True Positive: ", t_p)
    print("Detections - False Positive: ", f_p)
    print("Detections - False Negative ", f_n)
    print("Extra True Positive (Ignored as not max iou): ", ignored)


#     for key in true_positive.keys():
#         print("Found ", len(true_positive[key])," for ", key)
#         print(true_positive[key])
#         print("")

    if show_plot:
        l1 = []
        l2 = []
        for key in matched.keys():
            l1.append(matched[key]['gt_box'])
            l2.append(matched[key]['pred_box'])

        show_poly_anno(filename, [l1, l2], ["Matched Boxes: White: GT, Red: Pred"], False, (10, 10))
        show_poly_anno(filename, [gt_boxes, l2], ["GT Boxes", "Matched Pred Boxes"], True, (20, 15))

        l1 = []
        for key in not_matched.keys():
            l1.append(not_matched[key]['gt_box'])

        l2 = []
        for p_idx, box in enumerate(pred_boxes):
            if p_idx not in true_positive:
                l2.append(box)
                for g_idx, gt_box in enumerate(gt_boxes):
                    iou = get_poly_iou(gt_box, box[:-1])
                    if iou > iou_thresh:
                        print([*used])
                        print(p_idx, g_idx)
                        show_poly_anno(filename, [[gt_box], [box]], ["GT", "PRED"], True, (20, 15))
                        print("Something is wrong")

        show_poly_anno(filename, [l1, l2], ["False Negatives", "False Positives"], True, (20, 15))

        if len(true_positive.keys()) - len(matched) > 0:
            l1 = []
            l2 = []

            l3 = []
            for key in matched.keys():
                l3.append(matched[key]['indx'])

            for key in true_positive.keys():
                if key in l3:
                    l1.append(true_positive[key][0]['pred_box'])
                else:
                    print(true_positive[key])
                    l2.append(true_positive[key][0]['pred_box'])

            show_poly_anno(filename, [l1, l2], ["Ignored Boxes: White: GT, Red: Ignored"], False, (10, 10))


    return t_p, f_p, f_n, ignored
