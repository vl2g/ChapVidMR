import torch
import numpy as np
import torch.nn.functional as F

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from qd_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array, load_jsonl
import os

import os
import json
import torch


class QDDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        self.max_frames = 75
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1 / self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained QD-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        total_frames = len(video_feats)
        
        predictions = []
        for start_frame in range(0, total_frames, self.max_frames):
            end_frame = min(start_frame + self.max_frames, total_frames)
            clip_feats = video_feats[start_frame:end_frame]
            clip_predictions = self.process_clip(clip_feats, query_list, start_frame, total_frames)
            predictions.extend(clip_predictions)

        return predictions

    def process_clip(self, clip_feats, query_list, start_frame, total_frames):
        n_query = len(query_list)
        n_frames = len(clip_feats)
        tef_st = torch.arange(start_frame, start_frame + n_frames) / total_frames
        tef_ed = tef_st + 1.0 / total_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # Time encoding features
        clip_feats = torch.cat([clip_feats, tef], dim=1)
        clip_feats = clip_feats.unsqueeze(0).repeat(n_query, 1, 1)  # Repeat for each query
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        
        query_feats = self.feature_extractor.encode_text(query_list)
        query_feats, query_mask = pad_sequences_1d(query_feats, dtype=torch.float32, device=self.device)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        
        model_inputs = dict(
            src_vid=clip_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )
        
        outputs = self.model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0]  # foreground label is assumed to be 0
        pred_spans = outputs["pred_spans"]
        pred_spans = span_cxw_to_xx(pred_spans) * total_frames * self.clip_len  # Adjust spans to full video duration
        
        clip_predictions = []
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            filtered_results = []
            for span, sc in zip(spans, score):
                if sc > 0.5:
                    filtered_results.append([span[0].item(), span[1].item(), sc.item()])  # Convert tensor to list of floats
            filtered_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)[:2]  # Top-2 high score results
            clip_predictions.extend(filtered_results)
        
        return clip_predictions


def process_video(json_path, video_folder, output_folder, predictor, json_file_name):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    video_id = data['id']
    query = data['query']
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(video_folder, video_filename)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}. Skipping...")
        return
    
    predictions = predictor.localize_moment(video_path, [query])

    output_data = {
        'id': video_id,
        'query': query,
        'predictions': predictions
    }

    output_file = os.path.join(output_folder, f"{json_file_name}.json")
    with open(output_file, 'w') as out_file:
        json.dump(output_data, out_file, indent=4)  # Now it should work without serialization error

    print(f"Results saved for video {video_id} in {output_file}")

def main():
    json_folder = "path to ground truth json folder"
    video_folder = "path to videos"
    output_folder = "path to output json file"
    checkpoint_path = "path to checkpint"
    
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize the predictor
    predictor = QDDETRPredictor(ckpt_path=checkpoint_path, device='cuda')

    # Process each JSON file
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            process_video(json_path, video_folder, output_folder, predictor, json_file.split('.')[0])

if __name__ == "__main__":
    main()
