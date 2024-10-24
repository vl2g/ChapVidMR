import os
import json
import pdb
import time
import torch
import numpy as np
import argparse
import subprocess
from run_on_video import clip, vid2clip, txt2clip

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='./tmp')
parser.add_argument('--resume', type=str, default='./results/pt+ft/model_best.ckpt')
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument('--video_folder', type=str, required=True, help='Folder containing videos')
parser.add_argument('--query_folder', type=str, required=True, help='Folder containing JSON files for queries')
parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the results')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

#################################
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
    # pdb.set_trace()
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
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

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


    pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    pred_confidence = pred_logits
    
 
    top1_window = pred_windows[torch.argmax(pred_confidence)].tolist()
    top5_values, top5_indices = torch.topk(pred_confidence.flatten(), k=5)
    top5_windows = pred_windows[top5_indices].tolist()
    
   
    q_response = f"For query: {query}"

    mr_res =  " - ".join([convert_to_hms(int(i)) for i in top1_window])
    mr_response = f"The Top-1 interval is: {mr_res}"
    
    hl_res = convert_to_hms(torch.argmax(pred_saliency) * clip_len)
    hl_response = f"The Top-1 highlight is: {hl_res}"
    return '\n'.join([q_response, mr_response, hl_response])
    
def extract_vid(vid_path):
    vid_features = vid2clip(clip_model, vid_path, args.save_dir)
    return vid_features

def extract_txt(txt):
    txt_features = txt2clip(clip_model, txt, args.save_dir)
    return txt_features

def process_videos(video_folder, query_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for vid_file in os.listdir(video_folder):
        if vid_file.endswith('.mp4'):
            vid_id = os.path.splitext(vid_file)[0]
            vid_path = os.path.join(video_folder, vid_file)
            query_file = os.path.join(query_folder, f'{vid_id}.json')
            
            if os.path.exists(query_file):
                with open(query_file, 'r') as f:
                    query_data = json.load(f)
                query = query_data.get('query', '')
                
                if query:
                    result = forward(vtg_model, args.save_dir, vid_path, query)
                    output_file = os.path.join(output_folder, f'{vid_id}_result.txt')
                    with open(output_file, 'w') as f:
                        f.write(result)
                    print(f'Results saved for video {vid_file}')
                else:
                    print(f'No query found in {query_file}')
            else:
                print(f'Query file {query_file} not found')

if __name__ == '__main__':
    process_videos(args.video_folder, args.query_folder, args.output_folder)
