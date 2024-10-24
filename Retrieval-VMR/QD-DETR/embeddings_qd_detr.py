import json, csv, ast, traceback
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import os
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from qd_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array
import torch.nn.functional as F
import numpy as np
import pickle as pkl
yta = pickle.load(open('path-to-chapters.pkl-from-YT8m-dataset', 'rb'))



class QDDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained QD-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats) 
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames        
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained QDDETR only support video up " \
                            "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )
        
        outputs = self.model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)  
        scores = prob[..., 0] 
        pred_spans = outputs["pred_spans"]
        _saliency_scores = outputs["saliency_scores"].half()
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx], 
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx] 
            )
            predictions.append(cur_query_pred)
        
        return predictions

def run_example(video_paths, query_text):
    query_text_list = [ query_text]
    chap_predictions = []
    for vid_path in video_paths:
        prediction = qd_detr_predictor.localize_moment(
            video_path=vid_path, query_list=query_text_list)
        chap_predictions.append( prediction)

    chap_saliency_scores = []
    for pred in chap_predictions:
        chap_saliency_scores.append( sum(pred[0]['pred_saliency_scores'])/len(pred[0]['pred_saliency_scores']))
    return chap_saliency_scores

def chap_lis_retrieve( id):
    dur_info = {}
    if id in msdbk:
        meta = json.load(open(f'path-to-meta-data/{id}--info_minify.json'))
        chaps = [_['title'] for _ in meta['chapters']]
        for c in meta['chapters']:
            dur_info[ c['title']] = c['end_time'] - c['start_time']
    elif id in yt7mdbk:
        chaps = [_['label'] for _ in yta[id]['chapters']]
        for c in range( len(yta[id]['chapters'])):
            if c< len( yta[id]['chapters'])-1:
                dur_info[ yta[id]['chapters'][c]['label']] = yta[id]['chapters'][c+1]['time'] - yta[id]['chapters'][c]['time']
            else:
                dur_info[ yta[id]['chapters'][c]['label']] = yta[id]['duration'] - yta[id]['chapters'][c]['time']

    return chaps, dur_info

def tpc_fpc_fnc_calc(gt_chaps, retrieved_chaps):
    tpc, fpc, fnc = [], [], []
    for r_chap in retrieved_chaps:
        if r_chap in gt_chaps:
            tpc.append( r_chap)
        else:
            fpc.append( r_chap)
    for g_chap in gt_chaps:
        if g_chap not in retrieved_chaps:
            fnc.append( g_chap)
    return tpc, fpc, fnc

def duration_calc( l, dic):
    val = 0
    dic = json.loads( dic)
    for _ in l:
        try:
            val += dic[_]['duration']
        except:
            val = -100
            return val
    return val

if __name__ == "__main__":
    
    k = 2
    data_path = '../../data/sample_dataset.csv' 
    msdb = json.load( open( '../../assets/ms-data-backbone.json'))
    yt7mdb = json.load( open( '../../assets/yt7m-data-backbone.json'))
    msdbk = list( msdb.keys())
    yt7mdbk = list( yt7mdb.keys())
    dt_test = pd.read_csv(data_path) 

    vid_splits = 'path-to-vid-splits'
    vid_split_list = os.listdir( vid_splits)
    aud_splits = 'path-to-aud-splits'

    df = pd.read_csv(data_path)

    gt_read_error_counter = 0
    id_error_counter = 0
    vid_split_error = 0
    qd_detr_error = 0
    duration_calc_error = 0
    ave_iou = 0
    ave_precision = 0
    ave_recall = 0

    ckpt_path = "run_on_video/qd_detr_ckpt/model_best.ckpt"
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    qd_detr_predictor = QDDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device="cuda"
    )

    for _, point in tqdm(df.iterrows(), total=len(df)):
        id = point['id']
        query = point['query']
        chap_list_from_api, duration_info = chap_lis_retrieve( id)
        try:
            gt_chaps = ast.literal_eval( point['cleaned_chapters'])
        except:
            gt_read_error_counter +=1
            with open("chapter-read-errors.txt", "a") as f:
                f.write(f"{id} : {_}\n")
                traceback.print_exc(file=f) 
            continue

        chapter_list = []
        vid_paths = []
        vid_chap_names = []

       
        if id in msdbk:
            for chap_name in msdb[id]['chapter annotations']:
                chapter_list.append( chap_name)
        elif id in yt7mdbk:
            for chap_name in yt7mdb[id]['chapter annotations']:
                chapter_list.append( chap_name)
        else:
            id_error_counter +=1
            continue

        if id in vid_split_list:
            chap_splits = os.listdir( os.path.join( vid_splits, id))
            for chap in chap_splits:
                vid_paths.append( os.path.join( vid_splits, id, chap))
                vid_chap_names.append( chap)
        else:
            vid_split_error+=1
            continue

        try:    
            scores = run_example( vid_paths, query)
            qd_scores_dict = {}
            qd_scores_dict['scores'] = scores
            qd_scores_dict['vid_chap_names'] = vid_chap_names
            qd_scores_dict['chapter_list'] = chapter_list
            qd_scores_dict['gt_chaps'] = gt_chaps

            successful_file_save_name = f'path-to-scores-folder/{id}--{_}.pkl'
            with open(successful_file_save_name, "wb") as f:
                pkl.dump(qd_scores_dict, f)
        except Exception as e:
            print(e)
            qd_detr_error +=1
            retrieved_chaps = []

    print(id_error_counter)
    print( vid_split_error)
    print( gt_read_error_counter)
    print( duration_calc_error)
    print( qd_detr_error)