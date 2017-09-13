from __future__ import division

import pandas as pd
import numpy as np
import pickle

'''
Each chunk is 4 quarter-notes
'''
def save_data(path):
    z = pd.read_csv(path)
    num_files = z.file_index.max()
    num_notes = 98 - 34
    batch_size = 2400 // 32 * 4

    z.t0 = z.t0.apply(lambda x: x // 32)
    z.t1 = z.t1.apply(lambda x: x // 32)

    files = []

    for f_idx in range(0, num_files + 1):
        cur_file = z[z.file_index == f_idx]
        data = np.zeros(shape=(cur_file.t1.max(), num_notes), dtype=np.int8)
        print(f_idx, data.shape)
        for i in cur_file.as_matrix():
            #print(i[1], i[2], i[3]-36)
            data[i[1]:i[2], i[3]-34] = 1
        data = np.squeeze(np.array([np.array(data[i:i+batch_size]).flatten(order='C') for i in range(0, len(cur_file), batch_size)]))
        #data = np.asarray([[data[i+j] for j in range(batch_size)] for i in range(0, len(cur_file), batch_size)])
        #print(data.shape)
        #print(data[3])
        files.append(data)
        #break

    print(len(files))
    with open('./data/jsb_train.pkl', 'wb') as p:
        pickle.dump(files, p)

def load_data(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

#save_data('./data/SymbolicMusicMidiDataV1.0/jsb_train.csv')
