# Our Solution to the Kakao Arena - Melon Playlist Continuation Challenge (2nd Place on the Final Leaderboard)

This is our second-place solution for the Kakao Arena - Melon Playlist Continuation Challenge (Song nDCG: 0.321238, Tag nDCG: 0.507595). The task is to build a model that recommends songs relevant to playlists.

## Enviroment

  - Python 3.7.4
  - Intel Core i9-9960X+RAM 120GB Machine
  - The preprocessing step took 10-20 minitues, and the training and inference step took 2-3 hours (used 40GB memory).

### Dependencies

 - numpy/scipy
 - scikit-learn (0.22.2.post1)
 
## Model

 - This is a Neighbor-based CF model. The model estimates the similarity between songs included in a playlist u (item i) and songs not included (item j) as follows, and the most similar songs is recommended.
 
 ![equation1](https://user-images.githubusercontent.com/13177827/92404879-6cf6d100-f16f-11ea-9426-53a4c18e78ba.JPG)

- We used a discriminative reweighting technique to improve model performance by penalizing songs that do not represent their playlists (Zhu et al., 2018). For example, if some items in a playlist are unrelated to the other songs in the playlist, we apply panelty to these items, making them less likely to be recommended. To accomplish this, we generate a vector r_j for each item j that represents the similarity between each item j and the entire set of songs in the target playlist. Then, using r_j, which learns the playlist representativeness of each song j, we train an L2-regularized support vector classifier (SVC) that predicts y_j (1 if a playlist u contains an item j, 0 otherwise).
 
 ![equation2](https://user-images.githubusercontent.com/13177827/92404878-6bc5a400-f16f-11ea-88f5-afd636b3ac1f.JPG)

  - We use title-related keywords that appear 5 times or more in the dataset to reduce the cold start issue. Keywords are included in playlist items alongside other songs in the playlist.
 
## Dataset

 Download `train.json`, `val.json`, `test.json` into `res/` folder (https://arena.kakao.com/).

## How to run

 1. Run `python preprocess.py` to preprocess the dataset. 
 2. Run `python inference.py` to generate recommendations which are saved in `results.json`.

## Reference

Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking. In Proceedings of the ACM Recommender Systems Challenge 2018 (pp. 1-6). https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge
