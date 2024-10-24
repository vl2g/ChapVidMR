import os
import json
import numpy as np

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                try:
                    content = json.load(file)
                    if isinstance(content, dict):  # Now expecting a single dictionary object per file
                        data.append(content)
                    else:
                        print(f"Error: Expected a JSON object in file {filename}, but found a different type.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {filename}: {e}")
    return data

def process_ground_truth(gt_data):
    processed = []
    for item in gt_data:
        entry = {
            "qid": item["id"],
            "relevant_windows": [list(map(float, time_range.split('-'))) for time_range in item["finer-boundary"]]
        }
        processed.append(entry)
    return processed

def process_predictions(pred_data):
    processed = []
    for item in pred_data:
        entry = {
            "qid": item["id"],
            "pred_relevant_windows": [[float(pred[0]), float(pred[1]), float(pred[2])] for pred in item["predictions"]]
        }
        processed.append(entry)
    return processed

def compute_iou(pred, gt):
    intersection_start = max(pred[0], gt[0])
    intersection_end = min(pred[1], gt[1])
    intersection = max(0, intersection_end - intersection_start)
    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - intersection
    return intersection / union if union != 0 else 0

def evaluate_predictions(predictions, ground_truth, iou_thresholds):
    results = {str(th): {'tp': 0, 'fp': 0, 'fn': 0} for th in iou_thresholds}
    gt_matched = {str(th): set() for th in iou_thresholds}

    predictions_sorted = sorted(predictions, key=lambda x: -x[2])  # Sort by confidence score, descending

    for pred in predictions_sorted:
        for iou_threshold in iou_thresholds:
            matched = False
            for gt in ground_truth:
                if compute_iou(pred, gt) >= iou_threshold:
                    if not matched:
                        results[str(iou_threshold)]['tp'] += 1
                        gt_matched[str(iou_threshold)].add(tuple(gt))
                        matched = True
            if not matched:
                results[str(iou_threshold)]['fp'] += 1

    for iou_threshold in iou_thresholds:
        results[str(iou_threshold)]['fn'] = len(ground_truth) - len(gt_matched[str(iou_threshold)])

    # Calculating recall for each threshold
    recalls = {str(th): results[str(th)]['tp'] / (results[str(th)]['tp'] + results[str(th)]['fn']) if (results[str(th)]['tp'] + results[str(th)]['fn']) > 0 else 0 for th in iou_thresholds}

    # Calculating mean Average Precision (simplified version for demonstration)
    aps = {}
    for th in iou_thresholds:
        tp_cum = 0
        precision_accum = []
        for pred in predictions_sorted:
            if any(compute_iou(pred, gt) >= th for gt in ground_truth):
                tp_cum += 1
            if tp_cum > 0:
                precision = tp_cum / (predictions_sorted.index(pred) + 1)
                precision_accum.append(precision)
        aps[str(th)] = np.mean(precision_accum) if precision_accum else 0

    return {'recalls': recalls, 'mAPs': aps}

def main(gt_dir, pred_dir):
    gt_data = load_json_files(gt_dir)
    pred_data = load_json_files(pred_dir)

    processed_gt = process_ground_truth(gt_data)
    processed_pred = process_predictions(pred_data)

    iou_thresholds = [0.5, 0.7, 0.75]
    all_results = []

    for pred in processed_pred:
        for gt in processed_gt:
            if pred['qid'] == gt['qid']:
                metrics = evaluate_predictions(pred['pred_relevant_windows'], gt['relevant_windows'], iou_thresholds)
                all_results.append(metrics)

    if all_results:
        print("Evaluation Results:")
        # Calculate mean of each metric
        mean_metrics = {th: {'mean_recall': np.mean([res['recalls'][str(th)] for res in all_results]),
                             'mean_mAP': np.mean([res['mAPs'][str(th)] for res in all_results])}
                        for th in iou_thresholds}
        for th, metrics in mean_metrics.items():
            print(f"Mean metrics for IoU threshold {th}: Recall = {metrics['mean_recall']:.4f}, mAP = {metrics['mean_mAP']:.4f}")
    else:
        print("No matching data found for evaluation.")


if __name__ == '__main__':
    gt_directory = 'path to ground truth directory json files'
    pred_directory = 'path to predicted json files'
    main(gt_directory, pred_directory)