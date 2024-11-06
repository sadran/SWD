import copy
import pickle

import numpy as np
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_swd
from ..dataset import DatasetTemplate
import open3d as o3d
import pandas as pd


class SwdDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, mode='train', root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, mode=mode, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        if self.split == 'train':
            self.root_split_path = self.root_path / 'train'
        elif self.split == 'test':
            self.root_split_path = self.root_path / 'test'
        elif self.split == 'validation':
            self.root_split_path = self.root_path / 'validation'
        
        # loading samples id  from the imageset file 
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        dataframe_path = self.root_split_path / 'groundtruth.csv'
        dataframe = pd.read_csv(str(dataframe_path))
        self.dataframe = dataframe[dataframe['frame'].isin(self.sample_id_list)].set_index('frame')

        self.swd_infos = []
        self.include_swd_data(self.mode)

    def include_swd_data(self, mode):
        """
        if dataset info files exist, load them into self.swd_infos 
        """

        if self.logger is not None:
            self.logger.info('Loading SWD dataset')
        swd_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                swd_infos.extend(infos)

        self.swd_infos.extend(swd_infos)

        if self.logger is not None:
            self.logger.info('Total samples for SWD dataset: %d' % (len(swd_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        if self.split == 'train':
            self.root_split_path = self.root_path / 'train'
        elif self.split == 'test':
            self.root_split_path = self.root_path / 'test'
        elif self.split == 'validation':
            self.root_split_path = self.root_path / 'validation'

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        dataframe_path = self.root_split_path / 'groundtruth.csv'
        dataframe = pd.read_csv(str(dataframe_path))
        self.dataframe = dataframe[dataframe['frame'].isin(self.sample_id_list)].set_index('frame')

    def get_pcd(self, idx):
        pcd_file = self.root_split_path / 'point_clouds' / ('%s.pcd' % idx)
        assert pcd_file.exists()
        return np.array(o3d.io.read_point_cloud(str(pcd_file)).points, dtype=np.float32).reshape(-1, 3)

    def get_label(self, idx):
        if idx not in self.dataframe.index:
            print(f"Index {idx} not found in the data frame!!!!!!!")
            return -1
        label_df = self.dataframe.loc[idx]
        return object3d_swd.get_objects_from_label(label_df)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 3, 'pcd_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_x'] = np.array([obj.rx for obj in obj_list])
                #annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                #annotations['rotation_z'] = np.array([obj.rz for obj in obj_list])

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rot = annotations['rotation_x'][:num_objects]
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes = np.concatenate([loc, l, w, h, rot[..., np.newaxis]], axis=1)
                annotations['gt_boxes'] = gt_boxes

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_pcd(sample_idx)
                    corners_points = box_utils.boxes_to_corners_3d(gt_boxes)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_points[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)

        # for debuging 
        #for sample_id in sample_id_list:
        #    infos = process_single_scene(sample_id)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('swd_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['pcd_idx']
            points = self.get_pcd(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_x': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_x'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(loc)):
                        print('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                 loc[idx][0], loc[idx][1], loc[idx][2], dims[idx][0], dims[idx][1], dims[idx][2], single_pred_dict['rotation_x'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.swd_infos[0].keys():
            return None, {}

        from .swd_object_eval_python import eval as swd_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.swd_infos]
        ap_result_str, ap_dict = swd_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.swd_infos) * self.total_epochs

        return len(self.swd_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.swd_infos)

        info = copy.deepcopy(self.swd_infos[index])

        sample_idx = info['point_cloud']['pcd_idx']
        
        points = self.get_pcd(sample_idx)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_x']
            gt_names = annos['name']
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_swd_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = SwdDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, test_split, validation_split = 'train', 'test', 'validation'

    train_filename = save_path / ('swd_infos_%s.pkl' % train_split)
    test_filename = save_path / ('swd_infos_%s.pkl' % test_split)
    val_filename = save_path / ('swd_infos_%s.pkl' % validation_split)
    trainval_filename = save_path / 'swd_infos_trainval.pkl'
    
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    swd_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(swd_infos_train, f)
    print('swd info train file is saved to %s' % train_filename)

    dataset.set_split(validation_split)
    swd_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(swd_infos_val, f)
    print('swd info val file is saved to %s' % val_filename)

    dataset.set_split('test')
    swd_infos_test = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(swd_infos_test, f)
    print('swd info test file is saved to %s' % test_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(swd_infos_train + swd_infos_val, f)
    print('swd info trainval file is saved to %s' % trainval_filename)

    
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_swd_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2]), Loader=yaml.SafeLoader))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_swd_infos(
            dataset_cfg=dataset_cfg,
            class_names=['steering_wheel'],
            data_path=ROOT_DIR / 'data' / 'swd',
            save_path=ROOT_DIR / 'data' / 'swd'
        )
