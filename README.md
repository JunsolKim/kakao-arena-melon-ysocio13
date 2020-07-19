# Kakao Arena Melon Playlist Continuation Competition 연대사회13 Team Code

This repository contains the Python source code of our solutions to the Melon Playlist Continuation challenge.

## Running Environment and Required Computational Resources

We run our project under Python 3.7.4, on a Intel Core i9-9960X*16+120GB machine. The training part took almost 2 hours and less than 50GB memory. The inference part took almost 2 hours and less than 50GB memory.

### Dependencies

 - numpy
 - scipy
 - pandas
 - scikit-learn
 - lightgbm
 - (tqdm, q_tqdm 삭제 필요)

## Data Preparation
In order to replicate our final submissions to the Melon Playlist Continuation challenge, you first need to download train.json, test.json, and song_meta.json from https://arena.kakao.com/c/7/data. After downloading these files, put the files in a folder `res/`.

## Results Generation

 - After downloading the data in `res/`, you need to run `python 1_preprocess.py` to preprocess the data. 
 - Then, to train the models, you need to run `python 2_train.py`. (Generated models are (1) word2vec embedding model, (2) Keras MLP model to deal with coldstart issue, (3) lightgbm model which recommends based on nearest-neighbor collaborative filtering score and playlist/song metadata)
 - Finally, you need to run `python 3_inference.py` to get the final recommendation results, which will generate `results.json`.
