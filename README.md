# Our Solution to the Kakao Arena - Melon Playlist Continuation Challenge (2nd Position on the Final Leaderboard)

This is our 2nd position solution to Kakao Arena - Melon Playlist Continuation Challenge (Song nDCG: 0.321238, Tag nDCG: 0.507595). The task is to create the recommendation model that recommends songs that are relevant to playlists.

## Enviroment

  - Python 3.7.4
  - Intel Core i9-9960X+RAM 120GB Machine
  - The preprocessing step took 10-20 minitues, and the training and inference step took 2-3 hours (used 40GB memory).

### Dependencies

 - numpy/scipy
 - scikit-learn (0.22.2.post1)
 
## 모형 특징

 - This is a Neighbor-based CF model. The model estimates the similarity between each song i included in a playlist u and song j as follows, and the most similar song j is recommended.
 
 ![equation1](https://user-images.githubusercontent.com/13177827/92404879-6cf6d100-f16f-11ea-9426-53a4c18e78ba.JPG)

- To improve the model performance, we used discriminative reweighting technique that gives penality to songs that do not represent their playlists (Zhu et al., 2018). For instance, if some items contained in a playlist are not similar to the other songs in the playlist, we apply panelty to these items so that they are less likely to be recommended. To do so, for every item j, we create a vector r<sub>j</sub> that represents the similarity between each item j and the entire songs contained in the target playlist. Then we train a L2-regularized support vector classifier (SVC) that predicts y<sub>j</sub> (1 if a playlist u contains an item j, 0 otherwise) using r<sub>j</sub>, which learns the playlist representativeness of each song j. After that, we use this SVC model to decide whether or not to recommend the item j. 
 
 ![equation2](https://user-images.githubusercontent.com/13177827/92404878-6bc5a400-f16f-11ea-88f5-afd636b3ac1f.JPG)

  - To reduce the cold start issue, we use title-related keywords that occur 5 times or more in the dataset. Specifically, we include keywords in playlist items along with other songs in the playlist.
 
## Dataset

 Download `train.json`, `val.json`, `test.json` into `res/` folder (https://arena.kakao.com/).

## How to run

 1. Run `python preprocess.py` to preprocess the dataset. 
 2. Run `python inference.py` to generate recommendations which will be saved in `results.json`.

## Reference

Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking. In Proceedings of the ACM Recommender Systems Challenge 2018 (pp. 1-6). https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge
