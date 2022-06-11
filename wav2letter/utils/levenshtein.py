import numpy as np


def levenshtein_distance(ref, hyp):

    ref_len = len(ref)
    hyp_len = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if ref_len == 0:
        return hyp_len
    if hyp_len == 0:
        return ref_len

    if ref_len < hyp_len:
        ref, hyp = hyp, ref
        ref_len, hyp_len = hyp_len, ref_len

    # use O(min(ref_len, hyp_len)) space
    distance = np.zeros((2, hyp_len + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,hyp_len + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, ref_len+ 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, hyp_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[ref_len % 2][hyp_len]