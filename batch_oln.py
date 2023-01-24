from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import csv
import shutil
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.transforms.functional import to_pil_image
import torch
import matplotlib.pyplot as plt
import datetime as dt
import cv2





def predict_bboxes(model, frame, score_thresh):
    """
    model:
    frame:
    score_thresh:
    """
    result = inference_detector(model, frame)
    # print(result)

    labels = np.concatenate([
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ])
    bboxes = np.vstack(result)
    scores = bboxes[:, -1]

    if score_thresh > 0:
        inds = scores > score_thresh
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
    
    bboxes = bboxes.astype(np.int32)

    return result, bboxes, labels, scores





root = '/home/vsuciu/data/hdvila/videos/'
video_dir = os.path.join(root, 'debug')
video_fps = [os.path.join(video_dir, d) for d in os.listdir(video_dir) if d[-5:] != '.part']

# object detection model
config_file = 'configs/oln_box/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
object_detector = init_detector(config_file, checkpoint_file, device='cuda:0')
print(object_detector.CLASSES)


vid_reader = mmcv.VideoReader(video_fps[0])

frame_idx = [i for i in range(0, 100, 20)]
print(frame_idx)
frames = [vid_reader[i] for i in frame_idx]
print(frames[0].shape)

batch_frames = np.stack(frames, axis=0)

# this doesn't work with a fourth batch dimension because the preprocessing
# steps expect a 3D shape of (H, W, 3). Will have to try something else.
result, bboxes, labels, scores = predict_bboxes(
    object_detector,
    batch_frames[1],
    0.1
)

print(labels)

