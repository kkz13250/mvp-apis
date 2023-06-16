#tools/inference_camera.py CoBEVT

import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
import pickle
import logging
import numpy as np

from mvp.config import data_root
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.evaluate.detection import evaluate_single_vehicle
from mvp.visualize.general import draw_matplotlib


def test_sample():
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")
    case = dataset.get_case(1, tag="multi_vehicle")
    ego_id = min(list(case.keys()))

    perception = OpencoodPerception(fusion_method="early", model_name="voxelnet")
    result = perception.run(case, ego_id=ego_id)

    draw_matplotlib(result[ego_id]["lidar"], result[ego_id]["gt_bboxes"], result[ego_id]["pred_bboxes"], show=False, save="../tmp/a.png")


def test_normal_cases(dataset, perception):
    for case_id in range(210):
        case = dataset.get_case(case_id, tag="multi_frame")
        data_dir = os.path.join("../data/OPV2V/normal", "{:06d}".format(case_id))
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, "{}_{}.pkl".format(perception.model_name, perception.fusion_method))
        # if os.path.isfile(save_path):
        #     logging.warn("skip")
        #     continue
        
        report = []
        for frame_id in range(10):
            report.append({})
            if frame_id == 9:
                for victim_vehicle_id in list(case[9].keys()):
                    perception_result = perception.run(case[frame_id], ego_id=victim_vehicle_id)
                    report[frame_id][victim_vehicle_id] = {
                        "pred_bboxes": perception_result[victim_vehicle_id]["pred_bboxes"],
                        "pred_scores": perception_result[victim_vehicle_id]["pred_scores"],
                    }
        
        with open(save_path, 'wb') as f:
            pickle.dump(report, f)
        logging.warn("{} done".format(case_id))

test_sample()