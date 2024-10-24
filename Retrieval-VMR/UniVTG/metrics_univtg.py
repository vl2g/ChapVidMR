import pickle, os
import numpy as np
import csv, json, torch
import traceback

msdb = json.load( open( '../../assets/ms-data-backbone.json'))
yt7mdb = json.load( open( '../../assets/yt7m-data-backbone.json'))
msdbk = list( msdb.keys())
yt7mdbk = list( yt7mdb.keys())
yta = pickle.load(open('path-to-chapters.pkl-from-YT8m-dataset', 'rb'))
map_dir = pickle.load( open( 'path-todict-that-aligns-chapter-names-with-video-names', 'rb'))

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
    for _ in l:
        try:
            val += dic[_]
        except:
            return -100
    return val

def reverse_vid_mapper(id_, l):
    meta = map_dir[id_]
    vid_chaps = []
    for l_ in l:
        for k in list(meta.keys()):
            if l_==meta[k]:
                vid_chaps.append( k)
    return vid_chaps


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


duration_calc_error = 0
error_logged = 0
samples_processed = 0
ave_iou = 0
ave_precision = 0
ave_recall = 0

data_dir = '../../data/sample_dataset.csv' 
data_files = os.listdir(data_dir)

for file in data_files:
    id = file[:11]
    _, duration_info = chap_lis_retrieve(id)
    _path = os.path.join(data_dir, file)
    scores_dict = pickle.load( open(_path, "rb"))

    try:
        gt_chaps = scores_dict['gt_chaps']
        vid_chap_names = scores_dict['vid_chap_names']
        vid_scores = scores_dict['scores']
        top_two_sub_indices = np.argsort(vid_scores)[-2:][::-1]
        retrieved_chaps = [ vid_chap_names[top_two_sub_indices[0]], vid_chap_names[top_two_sub_indices[1]] ]
        corrected_chaps = reverse_vid_mapper(id, retrieved_chaps)
        tpc, fpc, fnc = tpc_fpc_fnc_calc( gt_chaps, corrected_chaps)

        tp = duration_calc( tpc, duration_info)
        fp = duration_calc( fpc, duration_info)
        fn = duration_calc( fnc, duration_info)

        if tp==-100 or fp==-100 or fn==-100:
            duration_calc_error+=1
            continue
        else:
            samples_processed +=1
            iou = tp/(tp+fp+fn)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            ave_iou +=iou
            ave_precision +=precision
            ave_recall +=recall
    except:
        error_logged += 1
        with open('logged-error.txt', 'a') as f:
            f.write( f'{id}: {_}\n')
            traceback.print_exc( file=f)
        continue


print( f'samples_prcessed: {samples_processed}')
print( f'ave_iou: {ave_iou/samples_processed}')
print( f'ave_precision: {ave_precision/samples_processed}')
print( f'ave_recall: {ave_recall/samples_processed}')

print(error_logged)
print( duration_calc_error)
