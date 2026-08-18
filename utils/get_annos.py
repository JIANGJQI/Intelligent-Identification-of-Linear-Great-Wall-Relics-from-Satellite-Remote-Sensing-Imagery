import json

import numpy as np

from utils.extract_masks import extract_masks
from pycocotools import mask as mask_utils


def get_annos(label_path, label_origin):
    if label_origin != "whu":
        with open(label_path, "r") as f:
            json_result = json.load(f)
            annos = json_result if label_origin == "sam2" else json_result["objects"]
    else:
        # get the mask, box and points from the label, and assign each object an id for tracking
        annos = extract_masks(label_path)
    print(f"object num: {len(annos)}")

    # format annos:
    format_annos = []
    for idx, item in enumerate(annos):
        anno = {}
        if label_origin == "whu":
            mask, (x, y, w, h), points = item.values()
            anno = {
                "points": points,
                "box": np.array([x, y, x + w, y + h]),
                "mask": mask,
            }
        else:
            box = item.get("bbox")
            anno = {"box": np.array(box)}
            if label_origin == "sam2":
                rle = item.get("rle")
                area = item.get("area")
                # ignore the mask with area less than 150
                if area < 150:
                    continue
                anno["mask"] = mask_utils.decode(rle) * 255

        format_annos.append(anno)

    return format_annos
