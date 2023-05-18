#tools/inference.py V2X-ViT

import os, sys, argparse
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
sys.path.append(os.path.join(root, "third_party/OpenCOOD"))
import numpy as np
from mvp.data.opv2v_dataset import OPV2VDataset
try:
    from mvp.perception.custom_opencood_perception import CustomOpencoodPerception
    use_perception = True
except:
    use_perception = False
from mvp.visualize.general import draw_matplotlib, draw_multi_vehicle_case


def run(case_id):
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/V2XSET"), mode="test")
    case = dataset.get_case(case_id, tag="multi_vehicle")
    ego_id = min(list(case.keys()))

    if use_perception:
        perception = CustomOpencoodPerception(dataset_name="v2xset")
        case = perception.run_multi_vehicle(case, ego_id)
        print(case)
        draw_multi_vehicle_case(case, ego_id=ego_id, pred_bboxes=case[0]["pred_bboxes"], gt_bboxes=case[0]["gt_bboxes"], mode="matplotlib", save="../tmp/c.png")
    else:
        draw_multi_vehicle_case(case, ego_id=ego_id, gt_bboxes=case[0]["gt_bboxes"], mode="open3d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
	parser.add_argument('case_id', type=int)
	args = parser.parse_args()
    run(args.case_id)