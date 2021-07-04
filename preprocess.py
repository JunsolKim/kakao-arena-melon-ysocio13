from operator import itemgetter
from scipy.sparse import coo_matrix, save_npz
import numpy as np
from collections import defaultdict

import os
import io
import distutils.dir_util
from collections import Counter
import json
import pickle 

def pickle_dump(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

def most_popular(playlists, col, topk_count):
    c = Counter()
    for doc in playlists:
        c.update(doc[col])
    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]

def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]

def title_into_words(title):
    word_index_list = []
    for word in all_word_list:
        if word[0] in title:
            word_index_list.append([word[0], title.index(word[0]), tags_mp_counter[word[0]] * -1])
    word_list = [word_index[0] for word_index in sorted(word_index_list, key=itemgetter(1,2))]
    word_list_popular = []
    i=0
    while i < len(word_list):
        same_words = [word_list[i]]
        for j in range(i+1, len(word_list)):
            if word_list[i] in word_list[j] or word_list[j] in word_list[i]:
                same_words.append(word_list[j])
            else:
                break
        i+=len(same_words)
        max_popularity = 0
        word_to_append = None
        for word in same_words:
            if tags_mp_counter[word] > max_popularity:
                word_to_append = word
                max_popularity = tags_mp_counter[word]
        if word_to_append is not None:
            word_list_popular.append(word_to_append)
    return word_list_popular

if not(os.path.isdir('data/')):
    os.makedirs('data/')

train = load_json('res/train.json')
val = load_json('res/val.json')
test = load_json('res/test.json')
train = train + val + test

print('rank popular songs/tags...')
data_by_yearmonth = defaultdict(list)
for q in train:
    try:
        data_by_yearmonth[q['updt_date'][0:4]].append(q)
    except:
        pass
    try:
        data_by_yearmonth[q['updt_date'][0:7]].append(q)
    except:
        pass
data_by_yearmonth = dict(data_by_yearmonth)

most_popular_results = {}
songs_mp_counter, most_popular_results['songs'] = most_popular(train, "songs", 200)
tags_mp_counter, most_popular_results['tags'] = most_popular(train, "tags", 50)
for y in data_by_yearmonth.keys():
    _, most_popular_results['songs' + y] = most_popular(data_by_yearmonth[y], "songs", 200)
    _, most_popular_results['tags' + y] = most_popular(data_by_yearmonth[y], "tags", 50)


print('split title into words...')
all_word_list = []
for t in tags_mp_counter.most_common(): # use tags as words dictionary
    if t[1] >= 5 and len(t[0]) > 1:
        all_word_list.append(t)
for q in train:
    q['title_words'] = title_into_words(q['plylst_title'])
for q in test:
    q['title_words'] = title_into_words(q['plylst_title'])

print('write train matrix...')
playlist_song_train_matrix = []
p_encode, s_encode, p_decode, s_decode = {}, {}, {}, {}
playlist_idx = 0
song_idx = 0
for q in train:
    if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
        p_encode[q['id']] = playlist_idx
        for s in q['songs']:
            if s not in s_encode.keys():
                s_encode[s] = song_idx
                song_idx += 1
            playlist_song_train_matrix.append([playlist_idx, s_encode[s]])
        playlist_idx += 1
s_decode['@tag_start_idx'] = song_idx
for q in train:
    if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
        for s in q['tags']:
            if s not in s_encode.keys():
                s_encode[s] = song_idx
                song_idx += 1
            playlist_song_train_matrix.append([p_encode[q['id']], s_encode[s]])
s_decode['@tag_title_start_idx'] = song_idx
for q in train:
    if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
        for s in q['title_words']:
            if '!title_' + str(s) not in s_encode.keys():
                s_encode['!title_' + str(s)] = song_idx
                song_idx += 1
            playlist_song_train_matrix.append([p_encode[q['id']], s_encode['!title_' + str(s)]])
playlist_song_train_matrix = np.array(playlist_song_train_matrix)
playlist_song_train_matrix = coo_matrix((np.ones(playlist_song_train_matrix.shape[0]),(playlist_song_train_matrix[:,0], playlist_song_train_matrix[:,1])),shape=(playlist_idx,song_idx))
save_npz('data/playlist_song_train_matrix.npz', playlist_song_train_matrix)
for s in s_encode.keys():
    s_decode[s_encode[s]] = s
pickle_dump(s_decode, 'data/song_label_decoder.pickle')
pickle_dump(p_encode, 'data/playlist_label_encoder.pickle')

title_words_mp_counter, _ = most_popular(train, "title_words", 50)

print('write test item indices...')
for q in test:
    if len(q['songs']) + len(q['tags']) + len(q['title_words']) >= 1:
        if np.mean([songs_mp_counter[i] for i in q['songs']] + [tags_mp_counter[i] for i in q['tags']] + [title_words_mp_counter[i] for i in q['title_words']]) > 1:
            items = [s_encode[s] for s in q['songs'] + q['tags']]
            try:
                for s in q['title_words']:
                    if '!title_' + str(s) in s_encode.keys():
                        items.append(s_encode['!title_' + str(s)])
            except KeyError:
                q['title_words'] = []
            q['items'] = items
    
    if 'songs'+q['updt_date'][0:7] in most_popular_results.keys():
        q['songs_mp'] = (remove_seen(q['songs'],most_popular_results['songs'+q['updt_date'][0:7]] + remove_seen(most_popular_results['songs'+q['updt_date'][0:7]], most_popular_results['songs'])))[:100]
        q['tags_mp'] = (remove_seen(q['tags'],most_popular_results['tags'+q['updt_date'][0:7]] + remove_seen(most_popular_results['tags'+q['updt_date'][0:7]], most_popular_results['tags'])))[:10]
    elif 'songs'+q['updt_date'][0:4] in most_popular_results.keys():
        q['songs_mp'] = (remove_seen(q['songs'],most_popular_results['songs'+q['updt_date'][0:4]] + remove_seen(most_popular_results['songs'+q['updt_date'][0:4]], most_popular_results['songs'])))[:100]
        q['tags_mp'] = (remove_seen(q['tags'],most_popular_results['tags'+q['updt_date'][0:4]] + remove_seen(most_popular_results['tags'+q['updt_date'][0:4]], most_popular_results['tags'])))[:10]
    else:
        q['songs_mp'] = remove_seen(q['songs'],most_popular_results['songs'][:100])
        q['tags_mp'] = remove_seen(q['tags'],most_popular_results['tags'][:10])

write_json(test, 'data/test_items.json')
