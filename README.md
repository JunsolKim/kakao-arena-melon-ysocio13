# Melon Playlist Continuation Challenge 연대사회13

This repository contains the Python source code of our solutions to the Kakao Arena Melon Playlist Continuation challenge.

## Running environment

We run our project under Python 3.7.4, on a Intel Core i9-9960X*16+120GB machine. The inference part took almost 5~6 hours and less than 40GB memory.

### Dependencies

 - numpy/scipy
 - scikit-learn (0.22.2.post1)

## Data

In order to replicate our final submissions to the Melon Playlist Continuation challenge, you first need to download `train.json`, `val.json`, `test.json` from https://arena.kakao.com/. After downloading these files, put the files in the folder `res/`. (Do not put the files in the folder `data/` ! This folder is for saving preprocessed data)

## How to generate the 파이널 리더보드 results

 1. After putting the aforementioned files in `res/`, you need to run `python preprocess.py` to get ready for the data. 
 2. Then you need to run `python inference.py` to get the final results, which will generate `results.json`.

## Reference

Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking. In Proceedings of the ACM Recommender Systems Challenge 2018 (pp. 1-6). https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge
