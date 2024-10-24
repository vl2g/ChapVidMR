import pandas as pd 
from tqdm import tqdm
import traceback
import csv
import os
import ast
import traceback
import json
import pdb
import time
import torch
import numpy as np
import argparse
import subprocess
from run_on_video import clip, vid2clip, txt2clip
import pickle as pkl

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='./tmp')
parser.add_argument('--resume', type=str, default='./results/pt+ft/model_best.ckpt')
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


model_version = "ViT-B/32"
output_feat_size = 512
clip_len = 2
overwrite = True
num_decoding_thread = 4
half_precision = False

clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)

import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse(args)
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model

vtg_model = load_model()

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(save_dir, vid_path, query):
    vid = extract_vid(vid_path)
    txt = extract_txt(query)
    vid = vid.astype(np.float32)
    txt = txt.astype(np.float32)
    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 2
    ctx_l = vid.shape[0]

    timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  
        vid = torch.cat([vid, tef], dim=1)  

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, save_dir, vid_path, query):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir, vid_path, query)
    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)
    
    
    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()
    return torch.mean( pred_saliency[0])

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
    # dic = json.loads( dic)
    for _ in l:
        try:
            val += dic[_]['duration']
        except:
            val = -100
            return val
    return val

    
def extract_vid(vid_path):
    vid_features = vid2clip(clip_model, vid_path, args.save_dir)
    return vid_features

def extract_txt(txt):
    txt_features = txt2clip(clip_model, txt, args.save_dir)
    return txt_features

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
yta = pkl.load(open('path-to-chapters.pkl-from-YT8m-dataset', 'rb'))

if __name__ == '__main__':
    msdb = json.load( open( '../../assets/ms-data-backbone.json'))
    yt7mdb = json.load( open( '../../assets/yt7m-data-backbone.json'))
    msdbk = list( msdb.keys())
    yt7mdbk = list( yt7mdb.keys())
    dt_test = pd.read_csv('../../data/sample_dataset.csv' ) 
    vid_splits = '/data1/user/video_project/dataset/splits/test/vid-splits'
    vid_split_list = os.listdir( vid_splits)

    
    gt_read_error_counter = 0
    id_error_counter = 0
    vid_split_error = 0
    duration_calc_error = 0
    error_logged = 0
    ave_iou = 0
    ave_precision = 0
    ave_recall = 0

    for _, point in tqdm(dt_test.iterrows(), total=len(dt_test), desc="Processing datapoints"):
        id = point['id']
        query = point['query']
        chap_list_from_api, duration_info = chap_lis_retrieve(id)
        try:
            gt_chaps = ast.literal_eval( point['cleaned_chapters'])
        except:
            gt_read_error_counter +=1
            with open("chapter-read-errors.txt", "a") as f:
                f.write(f"{id} : {_}\n")
                traceback.print_exc(file=f) 
            continue
        successful_file_save_name = f'path-to-test-scores-folder/{id}--{_}.pkl'
        if os.path.exists(successful_file_save_name):
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

        save_dict = {}

        
        try:
            vid_id = id
            prompt = query
            vid_scores = []
            for vid_path in vid_paths:
                sal_score = forward( vtg_model, args.save_dir, vid_path, prompt)
                vid_scores.append( sal_score)
                
            save_dict['scores'] = vid_scores
            save_dict['vid_chap_names'] = vid_chap_names
            save_dict['chap_list'] = chap_list_from_api
            save_dict['gt_chaps'] = gt_chaps


            successful_file_save_name = f'path-to-test-scores-folder/{id}--{_}.pkl'
            with open(successful_file_save_name, "wb") as f:
                pkl.dump(save_dict, f)
        except:
            error_logged +=1
            with open('logged-error.txt', 'a') as f:
                f.write( f'{id}: {_}\n')
                traceback.print_exc( file=f)
            continue

    print(id_error_counter)
    print(error_logged)
    print( vid_split_error)
    print( gt_read_error_counter)
    print( duration_calc_error)
