from sentence_transformers import SentenceTransformer, util
import pandas as pd
import ast
import json, os, pickle
from tqdm import tqdm

# load subtitles and visual captions files
msdb = json.load( open( '../assets/ms-data-backbone.json'))
yt7mdb = json.load( open( '../assets/yt7m-data-backbone.json'))
msdbk = list( msdb.keys())
yt7mdbk = list( yt7mdb.keys())


model = SentenceTransformer('msmarco-distilbert-base-v4')
data = pd.read_csv('../data/sample_dataset.csv') 

# variables to keep track of various exceptions and errors
counter = 0
lapses = 0

subtitle_corpus = []
_ = None

accuracy_score = 0
ave_iou = 0
ave_precision = 0
ave_recall = 0


for _, point in tqdm(data.iterrows(), total=len(data), desc="Processing datapoints"):
    tp, fp, fn = 0,0,0
    tpc, fpc, fnc = [], [], []
    query = point['query']
    try:
        gt_chaps = ast.literal_eval( point['cleaned_chapters'])
    except:
        counter +=1
        continue
    
    id = point['id']
    subtitle_corpus = []
    vis_cap_corpus = []
    chapter_list = []

    if id in msdbk:
        for chap_name in msdb[id]['chapter annotations']:
            subtitle_corpus.append( msdb[id]['chapter annotations'][ chap_name]['subtitles'])
            vis_cap_corpus.append( msdb[id]['chapter annotations'][chap_name]['visual_captions'])
            chapter_list.append( chap_name)
    elif id in yt7mdbk:
        for chap_name in yt7mdb[id]['chapter annotations']:
            subtitle_corpus.append( yt7mdb[id]['chapter annotations'][chap_name]['subtitles'])
            try:
                vis_cap_corpus.append( yt7mdb[id]['chapter annotations'][chap_name]['visual_captions'])
            except:
                vis_cap_corpus.append( "None")
            chapter_list.append( chap_name)

    
    unified_corpus = []
    for i in range( len( vis_cap_corpus)):
        if vis_cap_corpus[i] is not None and subtitle_corpus[i] is not None:
            unified_corpus.append( subtitle_corpus[i] + ' ' + vis_cap_corpus[i])
        elif subtitle_corpus[i] is None:
            unified_corpus.append( vis_cap_corpus[i])
        elif vis_cap_corpus[i] is None:
            unified_corpus.append( subtitle_corpus[i])
        else:
            unified_corpus.append( "")

    
    top_k = min( 5, len( subtitle_corpus))
    query_embedding = model.encode( query, convert_to_tensor=True)
    unified_corpus_embeddings = model.encode( unified_corpus, convert_to_tensor=True)

    save_dict = {}
    save_dict['query'] = query_embedding
    save_dict['unified_corpus'] = unified_corpus_embeddings
    save_dict['chapter_list'] = chapter_list
    save_dict['gt_chaps'] = gt_chaps
    
    successful_file_save_name = f'test_embeds/{id}--{_}.pkl'
    with open(successful_file_save_name, "wb") as f:
        pickle.dump(save_dict, f)

