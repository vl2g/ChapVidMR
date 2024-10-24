## ImageBind Experimentation

To run this experiment, one should have setup ImageBind locally. Please refer to their official repo [here](https://github.com/facebookresearch/ImageBind), for further instructions. Before executing the scripts, please make seperate directories for storing the embeddings and the video splits.

Once configured.
```
python3 embeddings_Ibind.py
python3 metrics_Ibind.py
```
The provided embedding generator code; takes both subtitles and visual captions as corpus. Make necessary edits to run the experiment using subtitles/visual-captions only.
