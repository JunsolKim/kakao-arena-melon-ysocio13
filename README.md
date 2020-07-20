# Kakao Arena Melon Playlist Continuation Challenge 연대사회13

This repository contains the Python source code of our solutions to the Melon Playlist Continuation challenge.

## Running environment

We run our project under Python 3.7.4, on a Intel Core i9-9960X*16+120GB machine. The training part took almost 2 hours and less than 50GB memory. The inference part took almost 2 hours and less than 50GB memory.

### Dependencies

 - numpy/scipy/pandas
 - scikit-learn
 - gensim
 - lightgbm
 - (tqdm, q_tqdm 삭제 필요)

## Data

In order to replicate our final submissions to the Melon Playlist Continuation challenge, you first need to download `train.json`, `val.json`, `test.json`, and `song_meta.json` from https://arena.kakao.com/. After downloading these files, put the files in a folder `res/`.

## How to generate the results

 1. After putting the aforementioned data in `res/`, you need to run `python preprocess.py` to preprocess the data. 
 2. Then, to train the models, you need to run `python train.py`. 
 - Generated models are (1) word2vec track+tag embedding model, (2) lightgbm model which uses neighbor-based similarity score (See Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking. In Proceedings of the ACM Recommender Systems Challenge 2018.) and playlist/song metadata.
 3. Finally, you need to run `python inference.py` to get the final results, which will generate `results.json`.
