# Copyright (c) OpenMMLab. All rights reserved.
import abc
import argparse
import os
import os.path as osp
import random as rd
import shutil
import string
import warnings
from collections import defaultdict
import random
import cv2
import mmcv
import numpy as np
import pdb
from tqdm import tqdm

random.seed(10708)

try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for NTURGB+D pose extraction.'
    )  # noqa: E501

import pdb

mmdet_root = '/home/ubuntu/code/mmdetection'
mmpose_root = '/home/ubuntu/code/mmpose'

args = abc.abstractproperty()
args.det_config = f'{mmdet_root}/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_score_thr = 0.5
args.pose_config = f'{mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def extract_frame(video_path):
    random_ids = []
    vid = cv2.VideoCapture(video_path)
    flag, frame = vid.read()
    cnt = 0
    frame_list = [frame, ]
    while flag:
        cnt += 1
        frame_list.append(frame)
        flag, frame = vid.read()
    if cnt > 30:
        random_ids = sorted(random.sample(list(range(len(frame_list))), k=30))
        frame_list = [frame_list[i] for i in random_ids]
    return frame_list, random_ids


def detection_inference(args, frame_lists):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_lists))
    for frame_list in frame_lists:
        result = inference_detector(model, frame_list)
        # TODO: add code for object detection here, also output some global features
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):

    def inside(box0, box1, thre=0.8):
        return intersection(box0, box1) / area(box0) > thre

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, thre=threshold):
        shape = [sum(bbox[:, -1] > thre) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    label = int(vid.split('/')[-1].split('A')[1][:3])
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))


def pose_inference(args, frame_lists, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_lists))
    num_frame = len(det_results)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)
    returned_features = np.zeros((num_person, num_frame, 17, 32), dtype = np.float32)
    features = None
    for i, (f, d) in enumerate(zip(frame_lists, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        res = inference_top_down_pose_model(model, f, d, outputs=['backbone'], format='xyxy')
        pose = res[0]
        if len(res) == 3:
            rf = res[2]
            for j in range(rf.shape[0]):
                returned_features[j, i] = rf[j]

        if len(res) > 1 and len(res[1]) > 0 and isinstance(res[1][0], dict) and  res[1][0].get(args.feature) is not None:
            feature = res[1][0][args.feature] # shapes (8, 17, H, W), 8=len(d), ie, len(pose), 17=num of keypoints in each bbox
        else:
            feature = None
        if features is None and feature is not None:
            features = np.zeros([num_person, num_frame, 17] + list(feature.shape[2: ]), dtype=np.float32) # In case we want to use features of other dimensions
        if feature is not None:
            for j, (item, f) in enumerate(zip(pose, feature)):
                kp[j, i] = item['keypoints']
                features[j, i] = f
        else:
            for j, item in enumerate(pose):
                kp[j, i] = item['keypoints']
        prog_bar.update()
    return kp, features, returned_features  # shapes (12, 173, 17, 3), 173=num of frames, 3 = (x, y, score), 17=num of keypoints in each bbox, 12=num_person


def ntu_pose_extraction(vid, skip_postproc=False):
    frame_lists, random_ids = extract_frame(vid)
    det_results = detection_inference(args, frame_lists)
    # if not skip_postproc:
    #     det_results = ntu_det_postproc(vid, det_results)
    pose_results, features, returned_features = pose_inference(args, frame_lists, det_results)
    anno = dict()
    anno['keypoint'] = pose_results[..., :2]
    anno['keypoint_score'] = pose_results[..., 2]
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = frame_lists[0].shape[:2]
    anno['original_shape'] = frame_lists[0].shape[:2]
    anno['total_frames'] = pose_results.shape[1]
    anno['label'] = osp.basename(vid).split('.')[1]
    anno['features'] = features
    anno['frame_ids'] = random_ids
    anno["returned_features"] = returned_features

    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--feature", type=str, default="keypoint_head.final_layer")
    parser.add_argument("--out_dir", type=str, default="keypoints")
    parser.add_argument("--data_dir", type=str, default='/home/ubuntu/data/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.skip_postproc = True
    args.feature = global_args.feature
    data_dir = global_args.data_dir
    vid_list = []
    if not os.path.exists(global_args.out_dir):
        os.mkdir(global_args.out_dir)
    for idx, dir_name in enumerate(os.listdir(data_dir)):
        abs_dir_name = os.path.join(data_dir, dir_name)
        if os.path.isdir(abs_dir_name):
            for filename in os.listdir(abs_dir_name):
                video = os.path.join(abs_dir_name,filename)
                output = os.path.join(global_args.out_dir, filename.replace(".mp4", ".pkl"))
                vid_list.append((video, output, dir_name, idx))
    vid_list.sort()
    for video, output, dir_name, idx in tqdm(vid_list):
        args.video = video
        args.output = output
        anno = ntu_pose_extraction(args.video, args.skip_postproc)
        anno['label'] = idx
        anno['label_name'] = dir_name
        mmcv.dump([anno], args.output)
        print(video)
