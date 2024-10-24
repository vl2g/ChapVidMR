from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import ast
import json, os, pickle
from tqdm import tqdm

yta = pickle.load(open('../assets/chapters.pkl', 'rb'))

# function to generate chapter-list and their duration info in real time
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

import difflib
# function to align corrupt chapter names with real chapter names
def correct_word(ideal_words, corrupted_word):
    min_distance = float('inf')
    closest_match = None
    multiple_min_distance = False
    for word in ideal_words:
        distance = sum(a != b for a, b in zip(corrupted_word, word))
        if distance < min_distance:
            min_distance = distance
            closest_match = word
            multiple_min_distance = False
        elif distance == min_distance:
            multiple_min_distance = True
    return closest_match if not multiple_min_distance else None

def name_aligner( corrupted_list, golden):
    return [correct_word(golden, w) for w in corrupted_list]

def duration_calc( l, dic):
    val = 0
    try:
        for _ in l:
            val += dic[_]
    except:
        return -100
    return val

# load subtitles and visual captions files
msdb = json.load( open( '../assets/ms-data-backbone.json'))
yt7mdb = json.load( open( '../assets/yt7m-data-backbone.json'))
msdbk = list( msdb.keys())
yt7mdbk = list( yt7mdb.keys())

# msmarco-distilbert-base-v4
model = SentenceTransformer('msmarco-distilbert-base-tas-b')
data = pd.read_csv('../data/sample_dataset.csv') 
score_folder = 'test_embs'
score_files = os.listdir( score_folder)

# counter to keep track of various errors and exceptions
gt_read_error_counter = 0
lapses = 0
subtitle_corpus = []
counter_ = 0

accuracy_score = 0
ave_iou = 0
ave_precision = 0
ave_recall = 0

for score_file in score_files:
    path = os.path.join( score_folder, score_file)
    id = score_file[:11]
    score_dict = pickle.load( open( path, 'rb'))

    chapter_list_from_api, dur_info = chap_lis_retrieve( id)
    
    query_embedding = score_dict['query']
    chapter_embeddings = score_dict['unified_corpus']
    gt_chaps = score_dict['gt_chaps']
    chapter_list = score_dict['chapter_list']

    cos_scores_subtitle_corpus = util.dot_score(query_embedding, chapter_embeddings)[0]
    top_results_subtitle = torch.topk(cos_scores_subtitle_corpus, k=2)

    tp, fp, fn = 0,0,0
    tpc, fpc, fnc = [], [], []

    k = 0
    retrieved_chaps = []
    for score, idx in zip(top_results_subtitle[0], top_results_subtitle[1]):
        if k==2:
            break
        retrieved_chaps.append( chapter_list[idx])
        k +=1

    retrieved_chaps = name_aligner( retrieved_chaps, chapter_list_from_api)

    for r_chap in retrieved_chaps:
        if r_chap in gt_chaps:
            tpc.append( r_chap)
        else:
            fpc.append( r_chap)
    for g_chap in gt_chaps:
        if g_chap not in retrieved_chaps:
            fnc.append( g_chap)
    tp = duration_calc( tpc, dur_info)
    fp = duration_calc( fpc, dur_info)
    fn = duration_calc( fnc, dur_info)
    if tp==-100 or fp==-100 or fn==-100:
        lapses +=1
        continue

    counter_+=1
    iou = tp/(tp+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    ave_iou = ave_iou + iou 
    ave_precision += precision
    ave_recall += recall

print( f'average iou: {ave_iou/( counter_)}')
print( f'average precision: {ave_precision/(counter_)}')
print( f'average recall: {ave_recall/(counter_)}')
print( counter_)