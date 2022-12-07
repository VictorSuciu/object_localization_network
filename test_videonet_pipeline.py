from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import csv
import shutil
import numpy as np
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torchvision.transforms.functional import to_pil_image


def make_dir(path, overwrite):
    if overwrite:
        try:
            shutil.rmtree(path)
        except:
            pass
    if not os.path.isdir(path):
        os.mkdir(path)


def classify_bbox(img, model, proproc, model_meta):
    """
    img:        PIL Image - image to classify
    model:      pytorch pretrained imagenet classifier
    preproc:    pytorch pretrained imagenet classifier preprocessing pipeline
    model_meta: pytorch pretrained imagenet classifier weights/metadata
    """
    # https://pytorch.org/vision/stable/models.html
    preproc_img = proproc(img).unsqueeze(0)
    pred = model(preproc_img).squeeze(0).softmax(0)
    class_id = pred.argmax().item()
    score = pred[class_id].item()
    class_name = model_meta.meta['categories'][class_id]

    return class_id, class_name, score


def predict_bboxes(model, frame, score_thresh):
    """
    model:
    frame:
    score_thresh:
    """
    result = inference_detector(model, frame)

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



def write_bbox_to_file(csv_writer, frame_idx, bbox, coco_label, coco_score, imgnet_label, imgnet_score):
    csv_writer.writerow({
        'frame_idx': frame_idx,
        'x1': bbox[0],
        'x2': bbox[1],
        'y1': bbox[2],
        'y2': bbox[3],
        'coco_label': coco_label,
        'coco_label_score': coco_score,
        'imagenet_label': imgnet_label,
        'imagenet_label_score': imgnet_score,
    })


def extract_bboxes(
    video_fp,
    bbox_fp,
    object_detector,
    bbox_classifier,
    classifier_meta,
    classifier_preproc,
    score_thresh=0.3,
    bbox_img_dir='',
    max_frames=None,
    sec_between_frames=0,
    sec_start_time=0):
    """
    video_fp:           str - file path to a single video
    bbox_fp:            str - file path to the bounding box csv file
    object_detector:    object detection model
    bbox_classifier:    pytorch pretrained imagenet classifier
    classifier_meta:    pytorch pretrained imagenet classifier weights/metadata
    classifier_preproc: pytorch pretrained imagenet classifier preprocessing pipeline
    score_thresh:       float [0, 1] - minimum objectness score to keep a bounding box. Boxes with a lower score are discarded
    bbox_img_dir:       str - directory path where to save each frame with bounding boxes overlayed
                        empty string causes the program to skip this visualization process (faster)
    max_frames:         int - maximum number of frames to process. None value causes the program to
                        continue until the end of the video
    sec_between_frames: float - number of seconds between each processed frame
    sec_start_time:     float - timestamp (in second) of the first frame to process

    saves the bounding boxes, imagenet classification, and coco classification of each frame to a csv file
    """
    visualize = len(bbox_img_dir) != 0
    if visualize:
        make_dir(bbox_img_dir, True)

    video = mmcv.VideoReader(video_fp)
    
    fps = video.fps
    num_frames = len(video)

    if max_frames == None:
        max_frames = num_frames

    num_saved_frames = 0
    count = 0
    frame_idx_saved = []
    next_timestamp = sec_start_time + sec_between_frames

    with open(bbox_fp, 'w') as bbox_file:
        frame_idx = -1
        csv_writer = csv.DictWriter(
            bbox_file,
            fieldnames=['frame_idx', 'x1', 'y1', 'x2', 'y2', 'coco_label', 'coco_label_score', 'imagenet_label', 'imagenet_label_score']
        )
        # iterate through each frame
        while num_saved_frames < max_frames:
            frame_idx += 1
            frame = next(video)
            cur_time = count / fps
            
            # only frames that are after the start time
            # and are the correct distance from the last frame are processed
            if cur_time >= sec_start_time and cur_time >= next_timestamp:
                print(f'{os.path.basename(video_fp)} frame {frame_idx}/{len(video)}')
                
                result, bboxes, labels, scores = predict_bboxes(object_detector, frame, score_thresh)
                
                if len(labels) == 0: # no objects detected
                    write_bbox_to_file(csv_writer, frame_idx, -np.ones((5,)), -1, -1, -1, -1)
                else:
                    # get image crops of each bounding box
                    patches = mmcv.imcrop(frame, bboxes[:, :4])
                    
                    for patch, bbox, coco_label, coco_score in zip(patches, bboxes, labels, scores):
                        pil_patch = to_pil_image(patch)
                        
                        # classify using imagenet classifier
                        imgnet_class_id, imgnet_class_name, imgnet_score = classify_bbox(pil_patch, bbox_classifier, classifier_preproc, classifier_meta)
                        
                        write_bbox_to_file(
                            csv_writer,
                            frame_idx,
                            bbox,
                            coco_label,
                            coco_score,
                            imgnet_class_name,
                            imgnet_score
                        )                

                if visualize:
                    object_detector.show_result(frame, result, score_thr=score_thresh, out_file=os.path.join(bbox_img_dir, f'frame_{frame_idx}.jpg'))

                num_saved_frames += 1
                frame_idx_saved.append(count)
                next_timestamp += sec_between_frames

            count += 1


video_dir = '../data/hdvila/videos/test_videos'
bbox_csv_dir = '../data/hdvila/videos/bboxes'

video_fps = [os.path.join(video_dir, d) for d in os.listdir(video_dir) if d[-5:] != '.part']

# object detection model
config_file = 'configs/oln_box/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
object_detector = init_detector(config_file, checkpoint_file, device='cuda:0')

# imagenet classifier model
# https://pytorch.org/vision/stable/models.html
classifier_meta = ResNet50_Weights.IMAGENET1K_V2
classifier_preproc = classifier_meta.transforms()
bbox_classifier = resnet50(weights=classifier_meta)
bbox_classifier.eval()

for vfp in video_fps:
    print('current video:', vfp)
    # visualization is slower. Only use for debugging
    visualization_dir = f'../data/hdvila/videos/{os.path.basename(vfp)}+visualized'

    extract_bboxes(
        vfp,
        os.path.join(bbox_csv_dir, f'{os.path.basename(vfp)}+bboxes.csv'),
        object_detector,
        bbox_classifier,
        classifier_meta,
        classifier_preproc,
        score_thresh=0.1,
        bbox_img_dir='',
        max_frames=None,
        sec_between_frames=0.3,
        sec_start_time=0
    )
    print()
