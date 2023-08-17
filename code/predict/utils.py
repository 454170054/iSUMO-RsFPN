import os.path
import uuid


def construct_dataset(seqs):
    uid = uuid.uuid4()
    file_name = str(uid) + '.csv'
    file_path = os.path.join('../../prediction/ds', file_name)
    f = open(file_path, 'w')
    f.write('protein_id,Sequence,Label\n')
    for i, seq in enumerate(seqs):
        l = len(seq)
        frags = []
        for pos in range(l):
            if 'K' == seq[pos]:
                prefix = 'X' * (19 - pos) + seq[0: pos] if pos < 19 else seq[pos - 19: pos]
                suffix = seq[pos + 1:] + 'X' * (19 - (l - pos - 1)) if (l - pos - 1) < 19 else seq[pos + 1: pos + 20]
                frag = prefix + 'K' + suffix
                frags.append(frag)
        for frag in frags:
            line = f'{i},{frag},0\n'
            f.write(line)
    return file_path



