## Sentence Bert Experimentation

To run this experimentation, one should have sentence-transformers configured locally. Please refer to their official repo [here](https://github.com/UKPLab/sentence-transformers).

To install Sentence-Transformers:
`pip install -U sentence-transformers`

Once configured.
```
python3 sbert-embedding-generator.py
python3 sbert-metrics-calculator.py
```
 The provided embedding generator code; takes both subtitles and visual captions as corpus. Make necessary edits to run the experiment using subtitles/visual-captions only.