import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import re


def minMaxScaler(data):
    mu = np.mean(data, axis=-1, keepdims=True)
    minV = np.min(data, axis=-1, keepdims=True)
    maxV = np.max(data, axis=-1, keepdims=True)
    return (data - mu) / (maxV - minV)


def AAIndex(filepath):
    encodings = []
    dataframe = pd.read_csv(filepath)
    sequences = list(dataframe['Sequence'])
    dataset = pd.read_excel(r"../../dataset/AAindex.slsx")
    count = 0
    for seq in sequences:
        seq = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(seq).upper())
        encode = encode_sequence(seq, dataset)
        encodings.append(encode)
    return np.array(encodings).astype(np.float32).transpose(0, 2, 1)


def encode_sequence(seq, dataset):
    '''
    encode gpcr sequence by AAindex
    :return: encoded sequence
    '''
    code = []
    meta_sequence = list(seq)
    meta_sequence_reshape = np.reshape(np.array(meta_sequence), (-1, 1))
    columns_name = list(dataset.columns)
    columns_name.remove('         Amino Acid\nIndex Type')
    columns_name.remove('Attribute')
    columns_name.remove('Source')
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoder.fit(np.reshape(columns_name, (-1, 1)))
    onehot_meta_sequence = onehot_encoder.transform(meta_sequence_reshape)
    for i in range(5):
        aaindex = dataset.iloc[i].values[1:22]
        aaindex = aaindex.astype(float)
        aaindex = minMaxScaler(aaindex)
        aaindex_encode = np.dot(aaindex.reshape((1, -1)), onehot_meta_sequence.T)
        aaindex_encode = aaindex_encode.flatten().tolist()
        code.append(aaindex_encode)
    return code

