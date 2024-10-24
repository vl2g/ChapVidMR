import pickle, os
import numpy as np
import csv, json

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
            val+= dic[_]
        except:
            return -100
    return val 

def reverse_aud_mapper( id, l):
    l_new = []
    for l_ in l:
        l_new.append( l_.replace('.mp3', '.mp4'))
    return reverse_vid_mapper( id, l_new)


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
            
results_path = 'path-to-embeddings'
data_path = '../../data/sample_dataset.csv' 
audio_lapses , video_lapses, sub_lapses = 0,0,0
audio_counter , video_counter, sub_counter = 0,0,0
audio_ave_iou, audio_ave_precision, audio_ave_recall = 0,0,0
video_ave_iou, video_ave_precision, video_ave_recall = 0,0,0
sub_ave_iou, sub_ave_precision, sub_ave_recall = 0,0,0
retrieval_counter = 0
agreement_flag = False
files_probed = 0
eval_index_list = []
files = list(os.listdir(results_path))
for _ in files:
    agreement_flag = True
    _path = os.path.join( results_path, _)
    _file = pickle.load( open(_path, 'rb'))
    id_  = _[:11]
    chapter_list, duration_inf0 = chap_lis_retrieve(  id_)
    files_probed +=1


    embeddings = _file['embeds']
    chapter_list_dumped = _file['chapter_list']
    query_emb = embeddings['text'][-1]
    text_embs = embeddings['text'][:-1]
    vid_embs = embeddings['vision']
    aud_embs = embeddings['audio']

    tq_scores = (query_emb @ text_embs.T).cpu().numpy()
    vq_scores = (query_emb @ vid_embs.T).cpu().numpy()
    aq_scores = (query_emb @ aud_embs.T).cpu().numpy()

    top_two_sub_indices = np.argsort(tq_scores)[-2:][::-1]
    top_two_vid_indices = np.argsort(vq_scores)[-2:][::-1]
    top_two_aud_indices = np.argsort(aq_scores)[-2:][::-1]

    try:
        subtitles_retrieved_chaps = [_file["chapter_list"][top_two_sub_indices[0]], _file["chapter_list"][top_two_sub_indices[1]] ]
        video_retrieved_chaps = [_file["vid_chapter_list"][top_two_vid_indices[0]], _file["vid_chapter_list"][top_two_vid_indices[1]]]
        audio_retrieved_chaps = [_file["aud_chapter_list"][top_two_aud_indices[0]], _file["aud_chapter_list"][top_two_aud_indices[1]]]
        gt_chaps = _file["gt_chaps"]
    except Exception as e:
        retrieval_counter +=1
        continue
    
    video_retrieved_chaps = reverse_vid_mapper(id_, video_retrieved_chaps)
    audio_retrieved_chaps = reverse_aud_mapper(id_, audio_retrieved_chaps)

    audio_tpc, audio_fpc, audio_fnc = tpc_fpc_fnc_calc(gt_chaps, audio_retrieved_chaps)
    video_tpc, video_fpc, video_fnc = tpc_fpc_fnc_calc(gt_chaps, video_retrieved_chaps)
    sub_tpc, sub_fpc, sub_fnc = tpc_fpc_fnc_calc(gt_chaps, subtitles_retrieved_chaps)


    video_tp = duration_calc( video_tpc, duration_inf0)
    video_fp = duration_calc( video_fpc, duration_inf0)
    video_fn = duration_calc( video_fnc, duration_inf0)
    if agreement_flag:
        if video_tp==-100 or video_fp==-100 or video_fn==-100:
            video_lapses +=1
            agreement_flag = False

    audio_tp = duration_calc( audio_tpc, duration_inf0)
    audio_fp = duration_calc( audio_fpc, duration_inf0)
    audio_fn = duration_calc( audio_fnc, duration_inf0)
    if agreement_flag:
        if audio_tp==-100 or audio_fp==-100 or audio_fn==-100:
            audio_lapses +=1
            agreement_flag = False


    sub_tp = duration_calc( sub_tpc, duration_inf0)
    sub_fp = duration_calc( sub_fpc, duration_inf0)
    sub_fn = duration_calc( sub_fnc, duration_inf0)
    if agreement_flag:
        if sub_tp==-100 or sub_fp==-100 or sub_fn==-100:
            sub_lapses +=1
            agreement_flag = False



    if agreement_flag:
        eval_index_list.append( _path.split('--')[-1])
        video_counter+=1
        iou = video_tp/(video_tp+video_fp+video_fn)
        precision = video_tp/(video_tp+video_fp)
        recall = video_tp/(video_tp+video_fn)
        video_ave_iou = video_ave_iou + iou 
        video_ave_precision += precision
        video_ave_recall += recall

        audio_counter+=1
        iou = audio_tp/(audio_tp+audio_fp+audio_fn)
        precision = audio_tp/(audio_tp+audio_fp)
        recall = audio_tp/(audio_tp+audio_fn)
        audio_ave_iou = audio_ave_iou + iou 
        audio_ave_precision += precision
        audio_ave_recall += recall

        sub_counter+=1
        iou = sub_tp/(sub_tp+sub_fp+sub_fn)
        precision = sub_tp/(sub_tp+sub_fp)
        recall = sub_tp/(sub_tp+sub_fn)
        sub_ave_iou = sub_ave_iou + iou 
        sub_ave_precision += precision
        sub_ave_recall += recall

print( audio_counter, video_counter, sub_counter)
print( audio_lapses, video_lapses, sub_lapses)
print( retrieval_counter)
print( files_probed)
print('\n')
print( 'audio average IOU: ',audio_ave_iou/audio_counter)
print( 'audio average precision: ',audio_ave_precision/audio_counter)
print( 'audio average recall: ',audio_ave_recall/audio_counter)


print('\n')
print( 'video average IOU: ',video_ave_iou/video_counter)
print( 'video average precision: ',video_ave_precision/video_counter)
print( 'video average recall: ',video_ave_recall/video_counter)

print('\n')
print( 'sub average IOU: ',sub_ave_iou/sub_counter)
print( 'sub average precision: ',sub_ave_precision/sub_counter)
print( 'sub average recall: ',sub_ave_recall/sub_counter)
