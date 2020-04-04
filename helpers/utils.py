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

def xywh_to_xyxy(boxes, r_numpy=False):
    boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return boxes if r_numpy else boxes.tolist()

def xyxy_to_poly(boxes, r_numpy=False):
    boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
    poly_boxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in boxes]

    return np.array(poly_boxes) if r_numpy else poly_boxes


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
