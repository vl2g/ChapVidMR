import warnings
warnings.filterwarnings("ignore")
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pandas as pd
import traceback 
import json, ast, os, pickle
from tqdm import tqdm

msdb = json.load( open( '../../assets/ms-data-backbone.json'))
yt7mdb = json.load( open( '../../assets/yt7m-data-backbone.json'))
msdbk = list( msdb.keys())
yt7mdbk = list( yt7mdb.keys())

dt_test = pd.read_csv('../../assets/ms-data-backbone.json') 

main_vid_dir = '../../assets/video-splits/' 
all_vids_list  = os.listdir(main_vid_dir)

main_aud_dir = '../../assets/audio-splits/'
all_auds_list = os.listdir(main_aud_dir)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

counter = 0
vid_issue_counter = 0
subtitle_skip = 0
vid_skip = 0
aud_skip = 0
complex_case_counter = 0
none_hook = 0
for _, point in tqdm(dt_test.iterrows(), total=len(dt_test), desc="Processing datapoints"):
    print( _)
    tp, fp, fn = 0,0,0
    tpc, fpc, fnc = [], [], []
    id = point['id']
    query = point['query']
    try:
        gt_chaps = ast.literal_eval( point['cleaned_chapters'])
    except:
        counter +=1
        with open("chapter-read-errors.txt", "a") as f:
            f.write(f"{id} : {_}\n")
            traceback.print_exc(file=f) 
        continue

    successful_file_save_name = f'path to embeddings/{id}--{_}.pkl'
    if os.path.exists( successful_file_save_name):
        continue

    subtitle_corpus = []
    vis_cap_corpus = []
    chapter_list = []
    vid_paths = []
    vid_chap_names = []
    aud_paths = []
    aud_chap_names = []

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
    else:
        subtitle_skip +=1
        continue

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

    if id in all_vids_list:
        chap_splits = os.listdir( os.path.join( main_vid_dir, id))
        for chap in chap_splits:
            vid_paths.append( os.path.join( main_vid_dir, id, chap))
            vid_chap_names.append( chap)
    else:
        vid_skip+=1
        continue


    if id in all_auds_list:
        chap_splits = os.listdir( os.path.join(main_aud_dir, id))
        for chap in chap_splits:
            aud_paths.append( os.path.join( main_aud_dir, id, chap))
            aud_chap_names.append( chap)
    else:
        aud_skip+=1
        continue
 
    subtitle_corpus = unified_corpus

    try:
        subtitle_corpus.append( query)
        subtitle_corpus = ["" if item is None else item for item in subtitle_corpus]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(subtitle_corpus, device),
            ModalityType.VISION: data.load_and_transform_video_data(vid_paths, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(aud_paths, device),
        }

    except Exception as e:
        vid_issue_counter+=1
        with open("embedding-gen-errors.txt", "a") as f:
            f.write("An error occurred: {}\n".format(str(e)))
            f.write(f'{id}\n\n')
            traceback.print_exc(file=f)
        continue

    with torch.no_grad():
        embeddings = model(inputs)
    
    query_emb = embeddings['text'][-1]
    pred_queryXsub = torch.softmax(embeddings[ModalityType.TEXT][:-1] @ query_emb.T, dim=-1)
    pred_queryXaud = torch.softmax(embeddings[ModalityType.AUDIO] @ query_emb.T, dim=-1)
    pred_queryXvid = torch.softmax(embeddings[ModalityType.VISION] @ query_emb.T, dim=-1)
    

    save_data = {
        'embeds' :embeddings,
        'chapter_list' :chapter_list,
        'vid_chapter_list' :vid_chap_names,
        'aud_chapter_list' :aud_chap_names,
        'gt_chaps' :gt_chaps
    }

    save_file_name = f'path to embeddings/{id}--{_}.pkl'
    with open( save_file_name, 'wb') as f:
        pickle.dump( save_data, f)

    if len(chapter_list)!=len( vid_chap_names) or len( chapter_list)!=len( aud_chap_names):
        complex_case_counter +=1


print( f'total samples :{_}')
print( f'issues in embedding generation :{vid_issue_counter}')
print( f'subtitle extraction error from the original dataset :{subtitle_skip}')
print( f'physical video discovery error : {vid_skip}')
print( f'physical audio discovery error : {aud_skip}')
print( f'gt chapter extraction error: {counter}')
print( 100*(vid_issue_counter+ subtitle_skip+ vid_skip + aud_skip+ counter)/_)

print( f'complex cases: {complex_case_counter}')