import numpy as np
import gc
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix, load_npz
from multiprocessing import Pool
from time import time
from sklearn.svm import LinearSVC

import os
import io
import distutils.dir_util
import json
import pickle 

def pickle_load(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./" + parent)
    with io.open("./" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj

def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]

def neighbor_based_cf(playlist_id):
    item_indices = test_item_indices[playlist_id]

    alpha, beta, theta = 0.9, 0.7, 0.99
    
    Cr = 0.4 + (100 - np.shape(item_indices)[0]) * 0.0055
    if Cr < 0.2:
        Cr = 0.2
    elif Cr > 1:
        Cr = 1
    
    song_playlist_train_matrix = lil_matrix(song_playlist_train_matrix_raw)
    song_playlist_train_matrix[:,p_encode[playlist_id]] = 0

    weight = song_playlist_train_matrix[item_indices, :].multiply(np.power(1e-1 + I_list, beta - 1)).sum(axis=0)
    weight = np.array(weight).flatten()
    weight = np.power(weight,theta)
    value = song_playlist_train_matrix[item_indices, :].multiply(weight)
    value = value.dot(song_playlist_train_matrix.transpose()) 
    I_song_i = np.power(1e-1+I_song[item_indices],-alpha)
    value = value.multiply(I_song_i.reshape((-1,1)))
    value = value.multiply(np.power(1e-1+I_song,alpha-1))
    value = csr_matrix(value)

    predictions = lil_matrix(value)
    label = np.zeros(song_playlist_train_matrix.shape[0])
    label[item_indices] = 1
    
    clf = LinearSVC(C=Cr,class_weight={0:1,1:1},tol=1e-6, dual = True, max_iter=360000)
    clf.fit(predictions.transpose(),label)
    predictions = clf.decision_function(predictions.transpose())
    predictions = np.argsort(np.array(predictions).flatten() - min(predictions))[::-1]

    return np.array(list(predictions[predictions < tag_start_idx][:400]) + list(predictions[(predictions >= tag_start_idx) & (predictions < tag_title_start_idx)][:100]))  

s_decode = pickle_load('data/song_label_decoder.pickle')
p_encode = pickle_load('data/playlist_label_encoder.pickle')
tag_start_idx = s_decode['@tag_start_idx']
tag_title_start_idx = s_decode['@tag_title_start_idx']

print("load train matrix...")
playlist_song_train_matrix = load_npz('data/playlist_song_train_matrix.npz')
song_playlist_train_matrix_raw = lil_matrix(playlist_song_train_matrix.transpose())

gc.collect()                                                                                                                                                                                              

I_song = np.array(song_playlist_train_matrix_raw.sum(axis=1)).flatten()
I_list = np.array(song_playlist_train_matrix_raw.sum(axis=0)).flatten()

print("load test data...")
test = load_json('data/test_items.json')
test_item_indices = dict()
test_playlist_id = []
for q in test:
    if 'items' in q.keys():
        test_item_indices[q['id']] = q['items']
        test_playlist_id.append(q['id'])

print("predictions begin...")
pool = Pool(23)
results = pool.map(neighbor_based_cf, test_playlist_id)
pool.close()
pool.join()

prediction_results = {}
for i in range(len(results)):
    prediction_results[test_playlist_id[i]] = {"songs": [s_decode[s] for s in results[i][:400]], "tags": [s_decode[s] for s in results[i][400:]]}

print("write results.json...")
answers = []
for q in test:
    if q['id'] in test_playlist_id:
        answers.append({'id': q['id'],
        'songs': remove_seen(q['songs'], prediction_results[q['id']]['songs'])[:100],
        'tags': remove_seen(q['tags'], prediction_results[q['id']]['tags'])[:10] })
    else:
        answers.append({'id': q['id'],
        'songs': remove_seen(q['songs'], q['songs_mp'])[:100],
        'tags': remove_seen(q['tags'], q['tags_mp'])[:10] })
    if len(answers[len(answers)-1]['songs']) < 100 or len(answers[len(answers)-1]['tags']) < 10:
        answers[len(answers)-1]['songs'] = (answers[len(answers)-1]['songs'] + remove_seen(q['songs'] + answers[len(answers)-1]['songs'], q['songs_mp']))[:100]
        answers[len(answers)-1]['tags'] = (answers[len(answers)-1]['tags'] + remove_seen(q['tags'] + answers[len(answers)-1]['tags'], q['tags_mp']))[:10]
write_json(answers, 'results.json')

print('end')
