# Melon Playlist Continuation Challenge 연대사회13

This repository contains the Python source code of our solutions to the Kakao Arena Melon Playlist Continuation challenge.

## Running environment

We run our project under Python 3.7.4, on a Intel Core i9-9960X*16+120GB machine. The preprocess and training parts took less than one hour. The inference part took almost 5~6 hours and less than 40GB memory.

### Dependencies

 - numpy/scipy/pandas
 - scikit-learn (0.22.2.post1)
 - gensim (3.8.1)

## Data

In order to replicate our final submissions to the Melon Playlist Continuation challenge, you first need to download `train.json`, `val.json`, `test.json` from https://arena.kakao.com/. After downloading these files, put the files in a folder `res/`.

## How to generate the results

 1. After putting the aforementioned data in `res/`, you need to run `python preprocess.py` to get ready for data. 
 2. Then, to train the playlist embedding model, you need to run `python train.py`. 
 3. Finally, you need to run `python inference.py` to get the final results, which will generate `results.json`.
