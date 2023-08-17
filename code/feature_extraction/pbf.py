import numpy as np
import tensorflow as tf

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

# Each sequence is added <START> and <END> tokens
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)


def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
    return [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(seq)]


def parse_seq(seq):
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))


def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    seqs_list = []
    for seq_tokens in map(tokenize_seq, seqs):
        seqs_list.append(seq_tokens)
    return seqs_list


def extract_embedding_features(seqs):
    seqs_list = tokenize_seqs(seqs)
    model = tf.keras.models.load_model('../../protein_bert_f/protein_bert')
    embedding_layer = model.layers[3]
    features = []
    for seq in seqs_list:
        feature = embedding_layer(np.array(seq)).numpy()
        features.append(feature.tolist())
    return np.array(features)


if __name__ == '__main__':
    seqs = ['VYLAREKKSHFIVALKVLFKSQIEKEGVEHQLRREIEIQ']
    print(extract_embedding_features(seqs))
