from mmdet.apis import init_detector, inference_detector
from mmdet.models import ResNet, FPN, RPNHead, StandardRoIHead
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import mmcv
import os
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import torch



def preproc_data(img_list, model):
    preproc_imgs = []
    for img in img_list:
        cfg = model.cfg
        device = next(model.parameters()).device  # model device

        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]

        # scatter to specified GPU
        data = scatter(data, [device])[0]
        data['img'][0].to(device)

        preproc_imgs.append(data)
        

    return preproc_imgs


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



# object detection model
debug = True
if debug:
    run_root = '/home/vsuciu/object_localization_network/'
else:
    run_root = ''

config_file = run_root + 'configs/oln_box/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = run_root + 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
object_detector = init_detector(config_file, checkpoint_file, device='cuda:0')


root = '/home/vsuciu/data/hdvila/videos/'
video_dir = os.path.join(root, 'debug')
video_fps = [os.path.join(video_dir, d) for d in os.listdir(video_dir) if d[-5:] != '.part']

print(object_detector.CLASSES)

vid_reader = mmcv.VideoReader(video_fps[0])

frame_idx = [i for i in range(0, 100, 20)]
# print(frame_idx)
frames = [vid_reader[i] for i in frame_idx]
# print(frames[0].shape)

preproc_datastruct = preproc_data(frames, object_detector)
preproc_frames = [preproc_datastruct[i]['img'][0][0] for i in range(5)]
batch_frames = torch.stack(preproc_frames, dim=0)
batch_frames = batch_frames.float()
# print(type(batch_frames))
# exit()

resnet_backbone = ResNet(
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
).to(device='cuda:0')

fpn_neck = FPN(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5
).to(device='cuda:0')

rpn_head = RPNHead(
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type='AnchorGenerator',
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='L1Loss', loss_weight=1.0)
).to(device='cuda:0')

roi_head = StandardRoIHead(
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=80,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=False)
)
print('\n\n\n')
print(batch_frames.shape)
print('\nResNet')
fpass = resnet_backbone.forward(batch_frames)
fpass_list = list(fpass)
print(len(fpass_list))
print([element.shape for element in fpass_list])

print('\nFPN')
fpass = fpn_neck.forward(fpass)
fpass_list = list(fpass)
print(len(fpass_list))
print([element.shape for element in fpass_list])

print('\nRPN')
fpass = rpn_head.forward(fpass)
print(type(fpass))
fpass_list = list(fpass)
print(len(fpass_list))

for e1 in fpass_list:
    print(type(e1), len(e1))
    for e2 in e1:
        print('   ', type(e2), e2.shape)

# print([(type(element), len(element)) for element in fpass_list])
# print([len(element[0]) for element in fpass_list])
# print(fpass_list)
# print([element.shape for element in fpass_list])

# this doesn't work with a fourth batch dimension because the preprocessing
# steps expect a 3D shape of (H, W, 3). Will have to try something else.
# result, bboxes, labels, scores = predict_bboxes(
#     object_detector,
#     batch_frames[1],
#     0.1
# )

# print(labels)

