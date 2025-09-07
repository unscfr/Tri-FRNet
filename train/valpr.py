from ultralytics import YOLO
import os
import numpy as np
from collections import defaultdict

# Load model
model = YOLO(
    r"/home/DSJ31/Documents/FR/FRYOLO/train/runs/VIS-8/VIS-yolo11n-500-0.34-0.198/weights/best.pt")

# Set paths
image_dir = r"/home/DSJ31/Documents/datasets/visdrone2019/VisDrone2019-DET-val/images/"
label_dir = r"/home/DSJ31/Documents/datasets/visdrone2019/VisDrone2019-DET-val/annotations/"
output_dir = r"/home/DSJ31/Documents/FR/FRYOLO/train/runs/VIS-8/VIS-yolo11n-500-0.34-0.198/valpr-output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]


def parse_visdrone_annotation(label_path):
    """Parse VisDrone annotation file and return ground truth boxes"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    gt_boxes = []
    for line in lines:
        data = line.strip().split(',')
        if len(data) >= 6:  # At least bbox and class info
            bbox = list(map(float, data[:4]))  # [x1,y1,w,h]
            class_id = int(data[5])
            gt_boxes.append({'bbox': bbox, 'class_id': class_id})
    return gt_boxes


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes in [x1,y1,w,h] format"""
    # Convert to [x1,y1,x2,y2] format
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def evaluate_detections(detections, gt_boxes, iou_threshold=0.5):
    """Evaluate detections against ground truth boxes"""
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Convert detections to [x1,y1,w,h] format
    det_boxes = []
    for det in detections.boxes:
        xywh = det.xywh.tolist()[0]  # [x_center,y_center,w,h]
        # Convert to [x1,y1,w,h] format
        x1 = xywh[0] - xywh[2] / 2
        y1 = xywh[1] - xywh[3] / 2
        det_boxes.append({'bbox': [x1, y1, xywh[2], xywh[3]], 'conf': det.conf.item()})

    # Match detections with ground truth
    matched_gt = set()
    matched_det = set()

    for i, det in enumerate(det_boxes):
        best_iou = 0
        best_gt_idx = -1

        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = calculate_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            matched_det.add(i)
        else:
            false_positives += 1

    false_negatives = len(gt_boxes) - len(matched_gt)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


# Process each image
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

    # Load ground truth
    gt_boxes = []
    if os.path.exists(label_path):
        gt_boxes = parse_visdrone_annotation(label_path)

    # Perform detection
    results = model(image_path)

    # Evaluate detections
    metrics = evaluate_detections(results[0], gt_boxes)

    # Generate output filename with metrics
    base_name, ext = os.path.splitext(image_file)
    output_name = f"{base_name}_p{metrics['precision']:.2f}_r{metrics['recall']:.2f}_f{metrics['f1_score']:.2f}{ext}"
    output_path = os.path.join(output_dir, output_name)

    # Save detection results
    results[0].save(filename=output_path)

    print(
        f"Processed {image_file} - Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1_score']:.2f}")
    print(f"Saved to: {output_path}")