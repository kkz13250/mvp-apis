import os, sys, argparse
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
sys.path.append(os.path.join(root, "third_party/OpenCOOD"))
import numpy as np
try:
    from mvp.perception.custom_opencood_perception import CustomOpencoodPerception
    use_perception = True
except Exception as e:
    print(e)
    use_perception = False
from mvp.visualize.general import draw_matplotlib, draw_multi_vehicle_case


def run(scenario_id, frame_id):
    print(use_perception)
    perception = CustomOpencoodPerception()
    #case = perception.get_frame_data(scenario_id, frame_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('s_id', type=int)
    parser.add_argument('f_id', type=int)
    args = parser.parse_args()
    run(args.s_id, args.f_id)