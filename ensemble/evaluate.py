import json
import os
import glob
import numpy as np
from collections import defaultdict

def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Map image_id to annotations
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Map image filename to image_id
    filename_to_id = {}
    for img in coco_data['images']:
        filename_to_id[img['file_name']] = img['id']
    
    # Map category_id to category name
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    return image_annotations, filename_to_id, category_names, coco_data


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def coco_to_xyxy(bbox):
    """Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def evaluate_predictions(pred_dir, annotation_file, iou_threshold=0.5):
    # Load ground truth
    image_annotations, filename_to_id, category_names, coco_data = load_coco_annotations(annotation_file)
    
    # Initialize metrics
    class_metrics = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0,
        'total_gt': 0, 'total_pred': 0,
        'iou_sum': 0.0, 'matched_count': 0
    })
    
    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    
    # Process each prediction file
    pred_files = glob.glob(os.path.join(pred_dir, "*_predictions.json"))
    
    for pred_file in pred_files:
        # Load predictions
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)
        
        # Get image filename from path
        image_path = pred_data['image_path']
        image_filename = os.path.basename(image_path)
        
        # Get ground truth for this image
        if image_filename not in filename_to_id:
            continue
        
        image_id = filename_to_id[image_filename]
        gt_annotations = image_annotations[image_id]
        
        total_images += 1
        
        # Convert ground truth to usable format
        gt_boxes = []
        for ann in gt_annotations:
            gt_boxes.append({
                'bbox': coco_to_xyxy(ann['bbox']),
                'category_id': ann['category_id'],
                'category_name': category_names[ann['category_id']],
                'matched': False
            })
        
        total_gt_boxes += len(gt_boxes)
        
        # Get predictions
        predictions = pred_data['detections']
        total_pred_boxes += len(predictions)
        
        # Match predictions to ground truth for each class
        for pred in predictions:
            pred_class = pred['class_name']
            pred_bbox = pred['bbox']
            pred_conf = pred['confidence']
            
            class_metrics[pred_class]['total_pred'] += 1
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt['category_name'] == pred_class and not gt['matched']:
                    iou = calculate_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                class_metrics[pred_class]['tp'] += 1
                class_metrics[pred_class]['iou_sum'] += best_iou
                class_metrics[pred_class]['matched_count'] += 1
                gt_boxes[best_gt_idx]['matched'] = True
            else:
                # False positive
                class_metrics[pred_class]['fp'] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt in gt_boxes:
            class_name = gt['category_name']
            class_metrics[class_name]['total_gt'] += 1
            if not gt['matched']:
                class_metrics[class_name]['fn'] += 1
    
    # Calculate final metrics
    results = {
        'dataset_info': {
            'total_images': total_images,
            'total_ground_truth_boxes': total_gt_boxes,
            'total_predicted_boxes': total_pred_boxes,
            'iou_threshold': iou_threshold
        },
        'per_class_metrics': {},
        'overall_metrics': {}
    }
    
    # Per-class metrics
    all_tp = 0
    all_fp = 0
    all_fn = 0
    precisions = []
    recalls = []
    f1_scores = []
    avg_ious = []
    
    for class_name in sorted(class_metrics.keys()):
        metrics = class_metrics[class_name]
        
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = metrics['iou_sum'] / metrics['matched_count'] if metrics['matched_count'] > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        if avg_iou > 0:
            avg_ious.append(avg_iou)
        
        results['per_class_metrics'][class_name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'average_iou': round(avg_iou, 4),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_ground_truth': metrics['total_gt'],
            'total_predictions': metrics['total_pred']
        }
    
    # Overall metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    results['overall_metrics'] = {
        'precision': round(overall_precision, 4),
        'recall': round(overall_recall, 4),
        'f1_score': round(overall_f1, 4),
        'macro_precision': round(np.mean(precisions), 4),
        'macro_recall': round(np.mean(recalls), 4),
        'macro_f1': round(np.mean(f1_scores), 4),
        'mean_iou': round(np.mean(avg_ious), 4) if avg_ious else 0.0,
        'detection_accuracy': round(all_tp / (all_tp + all_fp + all_fn), 4) if (all_tp + all_fp + all_fn) > 0 else 0.0
    }
    
    return results

def main():
    """Main evaluation function"""
    # Paths
    pred_dir = os.path.join("outputs", "img_predictions")
    annotation_file = os.path.join("..", "faster-rcnn", "test", "_annotations.coco.json")
    
    # Check if files exist
    if not os.path.exists(pred_dir):
        return
    
    if not os.path.exists(annotation_file):
        return
    
    results_50 = evaluate_predictions(pred_dir, annotation_file, iou_threshold=0.5)
    output_file_50 = os.path.join("outputs", "evaluation_metrics_iou50.json")
    with open(output_file_50, 'w') as f:
        json.dump(results_50, f, indent=2)
    
    results_75 = evaluate_predictions(pred_dir, annotation_file, iou_threshold=0.75)
    output_file_75 = os.path.join("outputs", "evaluation_metrics_iou75.json")
    with open(output_file_75, 'w') as f:
        json.dump(results_75, f, indent=2)

    results_95 = evaluate_predictions(pred_dir, annotation_file, iou_threshold=0.95)
    output_file_95 = os.path.join("outputs", "evaluation_metrics_iou95.json")
    with open(output_file_95, 'w') as f:
        json.dump(results_95, f, indent=2)

if __name__ == "__main__":
    main()
