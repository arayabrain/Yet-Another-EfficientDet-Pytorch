import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calc_confusion_matrix(coco_eval, score_threshold, iou_threshold):
    coco_eval.evaluate()

    scores = {
        (imgId, catId): [dt["score"] for dt in coco_eval._dts[(imgId, catId)]]
        for imgId in coco_eval.params.imgIds
        for catId in coco_eval.params.catIds
    }

    detection_detail = {}
    confusion_matrix = {"TP": 0, "FP": {"duplicate": 0, "non_duplicate": 0}, "FN": 0}

    for key in [
        (imgId, catId)
        for imgId in coco_eval.params.imgIds
        for catId in coco_eval.params.catIds
    ]:
        detection_detail[key] = {
            "tp": [],  # index of pred
            "fp_duplicate": [],  # index of pred
            "fp_non_duplicate": [],  # index of pred
            "fn": [],  # index of gt
        }

        filtered = filter(
            lambda iou_score: iou_score[2] >= score_threshold,
            zip(range(len(scores[key])), coco_eval.ious[key], scores[key]),
        )

        detected_indexes = set()

        for ind, ious, _ in filtered:
            if not any(map(lambda iou: iou > iou_threshold, ious)):
                confusion_matrix["FP"]["non_duplicate"] += 1
                detection_detail[key]["fp_non_duplicate"].append(
                    coco_eval._dts[key][ind]["bbox"]
                )
                continue

            found_index = np.where(ious == np.amax(ious))[0][0]
            if not found_index in detected_indexes:
                detected_indexes.add(found_index)
                confusion_matrix["TP"] += 1
                detection_detail[key]["tp"].append(coco_eval._dts[key][ind]["bbox"])
                continue
            else:
                confusion_matrix["FP"]["duplicate"] += 1
                detection_detail[key]["fp_duplicate"].append(
                    coco_eval._dts[key][ind]["bbox"]
                )
                continue

        gt_indexes = set(range(len(ious)))
        not_detected_indexes = gt_indexes - detected_indexes
        if not_detected_indexes:
            for ind in not_detected_indexes:
                detection_detail[key]["fn"].append(coco_eval._gts[key][ind]["bbox"])

    confusion_matrix["FN"] = len(coco_gt.getAnnIds()) - confusion_matrix["TP"]

    return confusion_matrix, detection_detail


def generate_imgs_with_result_bbox(
    coco_eval,
    detection_detail,
    test_img_folder,
    dst_img_folder,
    colors,
    brightness_adj_val=50,
):
    os.makedirs(dst_img_folder, exist_ok=True)

    for imgid in [imgid for imgid in coco_eval.params.imgIds]:
        imgid = int(imgid)
        file_name = coco_eval.cocoGt.loadImgs(ids=imgid)[0]["file_name"]
        img = cv2.imread(os.path.join(test_img_folder, file_name))

        # brightness adjustment
        img += brightness_adj_val

        for imgcat in [imgcat for imgcat in coco_eval.params.catIds]:
            imgcat = int(imgcat)
            detail = detection_detail[(imgid, imgcat)]
            for key in ("tp", "fp_duplicate", "fp_non_duplicate", "fn"):
                bboxes = detail[key]
                if bboxes:
                    for bbox, score in bboxes:
                        x1, y1, w, h = np.array(bbox).astype(int)
                        cv2.rectangle(
                            img, (x1, y1), (x1 + w, y1 + h), COLORS[key], thickness=2
                        )
                        cv2.putText(img, f"{score:0.2f}", (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3, COLORS[key], thickness=3)

        cv2.imwrite(os.path.join(dst_img_folder, file_name), img)


if __name__ == "__main__":
    score_threshold = 0.2
    iou_threshold = 10 ** -3
    COLORS = {
        "tp": (0, 255, 0),
        "fp_duplicate": (255, 255, 255),
        "fp_non_duplicate": (0, 153, 255),
        "fn": (0, 0, 255),
    }

    test_img_folder = "../datasets/fujiseal/dark_test"
    dst_img_folder = "../visualized_evaluation/fujiseal/dark_test/"
    gt_path = "../datasets/fujiseal/annotations/instances_dark_test.json"
    pred_path = "../results/dark_test_bbox_results.json"

    coco_gt = COCO(gt_path)
    coco_pred = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")

    confusion_matrix, detection_detail = calc_confusion_matrix(
        coco_eval=coco_eval,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    print(f"confusion matrix: {confusion_matrix}")

    generate_imgs_with_result_bbox(
        coco_eval=coco_eval,
        detection_detail=detection_detail,
        test_img_folder=test_img_folder,
        dst_img_folder=dst_img_folder,
        colors=COLORS,
        brightness_adj_val=50,
    )
