# Our Solutions to the Kakao Arena - Melon Playlist Continuation Challenge (2nd Position on the Final Leaderboard)

카카오 아레나 - Melon Playlist Continuation 대회에 참여한 연대사회13 팀의 Python 코드입니다. Neighbor-based collaborative filtering via discriminative reweighting 모형으로 파이널 리더보드 2위를 달성하였습니다 (곡 nDCG: 0.321238, 태그 nDCG: 0.507595).

## 개발 환경

Python 3.7.4가 설치된 Intel Core i9-9960X+RAM 120GB Machine을 사용하였습니다. Preprocess 소요 시간은 10-20분이며, Inference 소요 시간은 2-3시간이었습니다 (메모리 40GB 사용).

### Dependencies

 - numpy/scipy
 - scikit-learn (0.22.2.post1)
 
## 모형 특징

 - Neighbor-based CF 모형입니다. 각 플레이리스트 u에 포함된 item (곡/태그/제목 키워드) i와 새로운 item j 간에 유사도를 측정한 뒤, u와 가장 가까운 j들을 추천합니다. i와 j 간의 유사도 s<sub>ij</sub>는 다음과 같은 방식으로 측정하였습니다.
 
 ![equation1](https://user-images.githubusercontent.com/13177827/92404879-6cf6d100-f16f-11ea-9426-53a4c18e78ba.JPG)

- 추천 성능을 높이기 위해, 플레이리스트 대표성이 낮은 곡들에 패널티를 부여하는 discriminative reweighting 기법을 사용했습니다 (Zhu et al., 2018). 예를 들어, 플레이리스트 u에 포함된 어떤 item i가 u와 어울리지 않는다면 (u에 포함된 다른 item들과 유사하지 않다면), i에 penalty를 부여하여 i와 유사한 item들이 덜 추천되도록 합니다. 이를 위해, 모든 item j에 대하여, 플레이리스트 u에 포함된 각 item i과의 similarity를 나타낸 vector r<sub>j</sub>를 만듭니다. 이어서 각 r<sub>j</sub> vector로 y<sub>j</sub> (플레이리스트 u가 item j를 포함하는지 여부)를 예측하는 L2-regularized support vector classifier를 학습하는데, 이 때 각 item i의 플레이리스트 대표성이 학습됩니다. 다음으로, 학습된 SVC를 이용해 새로운 item j가 플레이리스트 u에 포함될지 여부를 예측합니다.
 
 ![equation2](https://user-images.githubusercontent.com/13177827/92404878-6bc5a400-f16f-11ea-88f5-afd636b3ac1f.JPG)

  - cold start issue를 해소하기 위해 제목 키워드를 item에 포함했습니다. 멜론 플레이리스트들에 5번 이상 등장한 태그들로 키워드 사전을 만든 뒤, 각 플레이리스트 제목에 포함된 키워드들을 item에 추가했습니다.

 
## 데이터

 https://arena.kakao.com/ 사이트의 `train.json`, `val.json`, `test.json` 데이터를 `res/` 폴더에 다운로드 받으세요.

## 결과 재현

 1. 데이터를 다운로드 받은 뒤, `python preprocess.py` 를 실행하여 데이터를 전처리하세요. 
 2. 다음으로 `python inference.py`를 실행하면, 추천 결과가 `results.json` 에 저장됩니다.

## Reference

Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking. In Proceedings of the ACM Recommender Systems Challenge 2018 (pp. 1-6). https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge
