import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


# @numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):

    det_size = len(dt_datas['name'])
    gt_size = len(gt_datas['name'])
    dt_scores = dt_datas['score']

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    for i in range(gt_size):
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION):
            fn += 1
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not assigned_detection[i]):
                fp += 1
        nstuff = 0
        fp -= nstuff
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


# @numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_datas,
                             dt_datas,
                             min_overlap,
                             thresholds):
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]
            gt_data = gt_datas[i]
            dt_data = dt_datas[i]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        
        if metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_x"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_x"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_x"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_x"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

def eval_class(gt_annos,
               dt_annos,
               metric,
               min_overlaps,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)

    precision = np.zeros(
        [num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_minoverlap, N_SAMPLE_PTS])
    
    for k, min_overlap in enumerate(min_overlaps):
        thresholdss = []
        for i in range(len(gt_annos)):
            ret = compute_statistics_jit(
                overlaps[i],
                gt_annos[i],
                dt_annos[i],
                min_overlap=min_overlap,
                thresh=0.0,
                compute_fp=False)
            _, _, _, _, thresholds = ret
            thresholdss += thresholds.tolist()
        thresholdss = np.array(thresholdss)
        thresholds = get_thresholds(thresholdss, len(gt_annos))
        thresholds = np.array(thresholds)
        pr = np.zeros([len(thresholds), 4])
        idx = 0
        for j, num_part in enumerate(split_parts):
            gt_datas_part = np.array([gt for gt in gt_annos[idx:idx + num_part]])
            dt_datas_part = np.array([dt for dt in dt_annos[idx:idx + num_part]])
            fused_compute_statistics(
                parted_overlaps[j],
                pr,
                total_gt_num[idx:idx + num_part],
                total_dt_num[idx:idx + num_part],
                gt_datas_part,
                dt_datas_part,
                min_overlap=min_overlap,
                thresholds=thresholds)
            idx += num_part
        for i in range(len(thresholds)):
            recall[k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
            precision[k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
        for i in range(len(thresholds)):
            precision[k, i] = np.max(
                precision[k, i:], axis=-1)
            recall[k, i] = np.max(recall[k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            PR_detail_dict=None):
    # min_overlaps: [num_minoverlap]

    ret = eval_class(gt_annos, dt_annos, 1, min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, 2, min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']

    return mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    min_overlaps = np.array([0.7, 0.5, 0.25])
    class_to_name = {
        0: 'steering_wheel'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    result = ''

    mAPbev, mAP3d, mAPbev_R40, mAP3d_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(len(min_overlaps)):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}:".format(min_overlaps[i])))
            result += print_str(f"bev  AP:{mAPbev[i]:.4f}")
            result += print_str(f"3d   AP:{mAP3d[i]:.4f}")

            result += print_str(f"{class_to_name[curcls]} "
                                "AP_R40@{:.2f}:".format(min_overlaps[i]))
            
            result += print_str((f"bev  AP:{mAPbev_R40[i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[i]:.4f}"))

            #if i == 0:
            ret_dict['%s_3d_R40@%f' % (class_to_name[curcls], min_overlaps[i])] = mAP3d_R40[i]
            ret_dict['%s_bev_R40@%f' % (class_to_name[curcls], min_overlaps[i])] = mAPbev_R40[i]
                
                
    return result, ret_dict
