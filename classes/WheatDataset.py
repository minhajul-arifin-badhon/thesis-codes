import sys
sys.path.append("..")

import os
import pandas as pd
import numpy as np
import mmcv
import itertools
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from helpers import utils
from collections import defaultdict


class WheatDataset:
    def __init__(self, path_json, path_images, file_names=[]):
        self._path_json = path_json
        self._path_images = path_images
        self._file_names = file_names

        self._images = {}  # id->image
        self._categories = {}  # id -> category
        self._annotations = {}  # file_name -> annotations

    def load_dataset(self):
        with open(self._path_json) as json_file:
            json_data = json.load(json_file)

            is_filter = self._file_names != []

            self._images = {}
            for image in json_data["images"]:
                if is_filter:
                    if image["file_name"] in self._file_names:
                        self._images[image["id"]] = image
                else:
                    self._images[image["id"]] = image

            self._categories = {}
            for category in json_data["categories"]:
                self._categories[category["id"]] = category

            if is_filter:
                json_data["annotations"] = list(
                    filter(
                        lambda x: x["image_id"] in self._images.keys(),
                        json_data["annotations"],
                    )
                )

            self._annotations = defaultdict(list)
            for annotation in json_data["annotations"]:
                filename = self._images[annotation["image_id"]]["file_name"]
                if is_filter:
                    if filename in self._file_names:
                        self._annotations[filename].append(annotation)
                else:
                    self._annotations[filename].append(annotation)

    def get_file_names(self, ids=[], is_numpy=True):
        items = []
        if ids:
            items = list(
                map(
                    lambda x: x["file_name"],
                    filter(lambda x: x["id"] in ids, self._images.values()),
                )
            )
        else:
            items = (
                list(map(lambda x: x["file_name"], self._images.values()))
                if not self._file_names
                else self._file_names
            )

        return np.array(items) if is_numpy else items

    def get_image_ids(self, filenames=[], is_numpy=True):
        items = []
        if filenames:
            items = list(
                filter(lambda x: x["file_name"] in filenames, self._images).keys()
            )
        else:
            items = list(self._images.keys())

        return np.array(items) if is_numpy else items

    def get_file_name_image_id_pair(self):
        items = {}
        for image in self._images.values():
            items[image["file_name"]] = image["id"]

        return items

    def get_anno_of_image_dict(self, filename):
        return self._annotations.get(filename, [])

    def get_anno_of_image_list(self, filename, keys=None, is_numpy=False):
        anns = self._annotations.get(filename, [])
        items = []
        if keys is None:
            items = [list(ann.values()) for ann in anns]
        else:
            for ann in anns:
                items.append([ann[key] for key in keys])

        return np.array(items) if is_numpy else items

    def get_bboxes_of_image(self, filename, r_format="XYXY"):
        xywh_boxes = self.get_anno_of_image_list(filename, ["bbox"], True)[:, 0].astype(
            int
        )
        if r_format == "XYXY":
            return utils.xywh_to_xyxy(xywh_boxes)

        if r_format == "XYP":
            return utils.xyxy_to_poly(utils.xywh_to_xyxy(xywh_boxes))

        return xywh_boxes

    def get_dataset_bboxes(self, r_format="XYXY"):
        all_bboxes = {}
        for image in self._images.values():
            all_bboxes[image["file_name"]] = self.get_bboxes_of_image(image["file_name"], r_format)
        return all_bboxes

    def get_dataset_dataframe(self):
        rows = []
        for image in self._images.values():
            boxes = self.get_bboxes_of_image(image["file_name"])
            for b in boxes:
                rows.append([image["file_name"], b[0], b[1], b[2], b[3]])

        df = pd.DataFrame(
            np.array(rows), columns=["filename", "xmin", "ymin", "xmax", "ymax"]
        )
        return df

    def get_dataset_stat(self, images=None):
        df = self.get_dataset_dataframe()

        if images:
            df = df[df["filename"].isin(images)]

        df_stat = (
            df.groupby(["filename"])
            .size()
            .reset_index(name="no_of_anno")
            .sort_values(by=["filename"])
        )
        return df_stat

    def save_as_csv(self, csv_path):
        df = self.get_dataset_dataframe()
        df.to_csv(csv_path)

    def save_as_coco_json(self, path_json):
        data_dict = {
            "images": list(self._images.values()),
            "categories": list(self._categories.values()),
            "annotations": list(
                itertools.chain.from_iterable(self._annotations.values())
            ),
        }
        mmcv.dump(data_dict, path_json)

    def save_images(self, path_source, path_target, remove_target_contents=True):
        filenames = self.get_file_names()

        if not os.path.exists(path_source):
            print("Source folder does not exist.")
            return

        if not os.path.exists(path_target):
            os.mkdir(path_target)

        if remove_target_contents:
            for filename in os.listdir(path_target):
                os.remove(os.path.join(path_target, filename))

        for filename in os.listdir(path_source):
            if filename in filenames:
                shutil.copy(
                    os.path.join(path_source, filename),
                    os.path.join(path_target, filename),
                )

    def display_annotations(self, filename):
        boxes = self.get_bboxes_of_image(filename, r_format="XYP")
        utils.show_poly_anno(
            self._path_images + filename, [boxes], ["Ground Truths: " + filename]
        )
