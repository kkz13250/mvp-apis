from concurrent.futures import process
import os, sys
import numpy as np
from collections import OrderedDict
import torch
import math
import copy
import random
import logging
import torch
import torch.nn.functional as F
# from pytorch3d.ops import box3d_overlap
import cv2

import opencood
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
import opencood.data_utils.datasets
from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import torch_tensor_to_numpy

from .perception import Perception
from mvp.config import third_party_root, data_root
from mvp.tools.iou import iou3d
from mvp.evaluate.detection import iou3d_batch
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor


class CobevtOpencoodPerception(Perception):
    def __init__(self, fusion_method="intermediate", model_name="pointpillar", dynamic=True):
        super().__init__()
        assert(model_name in ["pointpillar"])
        assert(fusion_method in ["intermediate"])
        self.name = "{}_{}".format(model_name, fusion_method)
        self.devices = "cuda:0"
        self.model_name = model_name
        self.fusion_method = fusion_method
        self.root = os.path.join(third_party_root, "OpenCOOD")
        if dynmaic:
            self.model_dir = os.path.join(self.root, "Models/cobevt")
        else:
            self.model_dir = os.path.join(self.root, "Models/cobevt_static")
        self.config_file = os.path.join(self.model_dir, "config.yaml")
        self.preprocessors = {
            #"early": self.early_preprocess,
            "intermediate": self.intermediate_preprocess,
            #"late": self.late_preprocess,
        }
        self.inference_processors = {
            #"early": inference_utils.inference_early_fusion,
            "intermediate": inference_utils.inference_intermediate_fusion,
            #"late": inference_utils.inference_late_fusion,
        }

        hypes = yaml_utils.load_yaml(self.config_file, None)
        hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
        hypes["validate_dir"] = os.path.join(data_root, "OPV2V/validate")
        self.dataset = build_dataset(hypes, visualize=False, train=False)
        self.model = train_utils.create_model(hypes)
        # we assume gpu is necessary
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, self.model, _ = train_utils.load_saved_model(self.model_dir, self.model)
        self.model.eval()

    def run(self, multi_vehicle_case, ego_id):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            pred_box_tensor, pred_score, gt_box_tensor = \
                self.inference_processors[self.fusion_method](batch_data,
                                                              self.model,
                                                              self.dataset)
        pred_bboxes = pred_box_tensor.cpu().numpy()
        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
        pred_scores = pred_score.cpu().numpy()
        return pred_bboxes, pred_scores
    
    def run_multi_vehicle(self, multi_vehicle_case, ego_id):
        pred_bboxes, pred_scores = self.run(multi_vehicle_case, ego_id)
        if pred_bboxes.shape[0] == 0:
            multi_vehicle_case[ego_id]["pred_bboxes"] = np.array([])
            multi_vehicle_case[ego_id]["pred_scores"] = np.array([])
        else:
            multi_vehicle_case[ego_id]["pred_bboxes"] = pred_bboxes
            multi_vehicle_case[ego_id]["pred_scores"] = pred_scores
        return multi_vehicle_case

    def retrieve_base_data(self, multi_vehicle_case, ego_id):
        data = OrderedDict()
        for vehicle_id, vehicle_data in multi_vehicle_case.items():
            data[vehicle_id] = OrderedDict()
            data[vehicle_id]['ego'] = (vehicle_id == ego_id)
            if "params" in vehicle_data:
                data[vehicle_id]['params'] = vehicle_data["params"]
            else:
                data[vehicle_id]['params'] = {
                    "lidar_pose": vehicle_data["lidar_pose"],
                    "vehicles": {},
                }
            if self.model_name in ["pointpillar", "emp"]:
                data[vehicle_id]['lidar_np'] = vehicle_data["lidar"].astype(np.float32)
                data[vehicle_id]['lidar_np'][:,3] = 1
            else:
                data[vehicle_id]['lidar_np'] = vehicle_data["lidar"][:,:4].astype(np.float32)
        return data

    def early_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_lidar_pose = base_data_dict[ego_id]["params"]['lidar_pose']

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)
            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.dataset.params['postprocess'][
                                                         'order']
                                                     )
        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.dataset.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.dataset.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'label_dict': label_dict})

        return processed_data_dict

    def intermediate_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.dataset.get_pairwise_transformation(base_data_dict,
                                             self.dataset.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []

        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.dataset.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.dataset.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)
        
        velocity = velocity + (self.dataset.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.dataset.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.dataset.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None],(self.dataset.max_cav - len(
                                               spatial_correction_matrix),1,1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                   padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'spatial_correction_matrix': spatial_correction_matrix,
             'pairwise_t_matrix': pairwise_t_matrix})

        return processed_data_dict

    def late_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        reformat_data_dict = self.dataset.get_item_test(base_data_dict)

        return reformat_data_dict

    def points_to_voxel_torch(self, pcd):
        # https://github.com/DerrickXuNu/OpenCOOD/blob/main/opencood/data_utils/pre_processor/voxel_preprocessor.py
        # full_mean = False
        # block_filtering = False
        data_dict = {}
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        max_points_per_voxel = self.dataset.pre_processor.params["args"]["max_points_per_voxel"]

        voxel_coords = torch.floor((pcd[:, :3] - 
                torch.tensor(lidar_range[:3]).to(self.device)
            ) / torch.tensor(voxel_size).to(self.device)).int()

        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = torch.unique(voxel_coords, dim=0,
                                                           return_inverse=True,
                                                           return_counts=True)
        
        voxel_features = torch.zeros((len(voxel_coords), max_points_per_voxel, 4), dtype=torch.float32).to(self.device)

        for i in range(len(voxel_coords)):
            pts = pcd[inv_ind == i]
            if voxel_counts[i] > max_points_per_voxel:
                pts = pts[:max_points_per_voxel, :]
                voxel_counts[i] = max_points_per_voxel

            voxel_features[i, :pts.shape[0], :] = pts

        data_dict['voxel_features'] = voxel_features
        data_dict['voxel_coords'] = voxel_coords
        data_dict['voxel_num_points'] = voxel_counts

        return data_dict

    def point_to_voxel_index(self, point):
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        voxel_index = (np.floor(point[:3] - lidar_range[:3]) / voxel_size).astype(int)
        return voxel_index
