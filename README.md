# Our Solutions to the Kakao Arena - Melon Playlist Continuation Challenge

  - This repository contains the Python source code of our solutions to the RecSys 2018 Kakao Arena - Melon Playlist Continuation Challenge (2nd Position on the Final Leaderboard).
  - 카카오 아레나 3회 대회인 Melon Playlist Continuation에 참여한 연대사회13 팀의 Python 코드입니다. 파이널 리더보드 2위를 달성하였습니다 (곡 nDCG: 0.321238 (2위), 태그 nDCG: 0.507595 (6위)).

## 실행 환경

Python 3.7.4이 설치된, Intel Core i9-9960X+RAM 120GB Machine에서 실행하였습니다. Preprocess 소요 시간은 10~20분이며, Inference 소요 시간은 2~3시간입니다 (메모리 40GB 사용).

### Dependencies

 - numpy/scipy
 - scikit-learn (0.22.2.post1)
 
## 모델 소개

 - Neighbor-based collaborative filtering 모델입니다. 각 플레이리스트와 새로운 곡/태그 간에 거리를 측정한 뒤, 거리가 가장 가까운 순서대로 새로운 곡/태그를 추천합니다. 플레이리스트 i와 곡/태그 j 간의 거리는 아래와 같은 방식으로 측정됩니다.
  
 - 거리를 측정할 때 "제목에 포함된 단어"도 사용한 이유는, 곡/태그가 주어지지 않은 플레이리스트의 cold start issue를 해소하기 위함입니다. 또한 Neighbor-based 모델이 적절하다고 생각한 이유는, 멜론 플레이리스트들이 하나의 플레이리스트 안에 다양한 주제/취향의 곡을 포함하기보다 한 가지 주제/취향에 집중하는 경향이 있다고 보았기 때문입니다. User 취향의 복잡성을 고려하는 추천 모델보다는, 단순한 Neighbor-based 모델이 더 높은 성능을 보일 것이라 보았습니다. 
 - 추천 성능을 높이기 위해 discriminative reweighting 기법을 사용했습니다 (Zhu et al., 2018). 구체적으로, Linear Support Vector Classifier를 통해 대표성이 낮은 곡들에 penalty를 부여하는 reweighting model을 학습했습니다. 플레이리스트에 포함된 곡들 가운데, 어떤 곡들은 플레이리스트의 특성을 잘 반영하지 못할 수도 있습니다. 예를 들어, 발라드 곡 중심의 플레이리스트에 소수의 힙합 곡들이 포함될 수도 있습니다. 만일 힙합 곡들과 가까운 곡들이 플레이리스트에 포함될 가능성이 낮은 것으로 확인될 경우, 힙합 곡들에 penalty를 부여하여 가까운 곡들이 덜 추천되도록 합니다. 
 
## 데이터

 https://arena.kakao.com/ 사이트의 `train.json`, `val.json`, `test.json` 데이터를 `res/` 폴더에 다운로드 받으세요.

## 결과 재현하기

 1. 데이터를 다운로드 받은 뒤, `python preprocess.py` 를 실행하여 데이터를 전처리하세요. 
 2. 다음으로, `python inference.py`를 실행하면 추천 결과가 `results.json` 에 저장됩니다.

## Reference

Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking. In Proceedings of the ACM Recommender Systems Challenge 2018 (pp. 1-6). https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge
