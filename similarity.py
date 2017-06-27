#!/usr/bin/env python3
from scipy.sparse import csr_matrix
from numpy import sqrt
from datetime import datetime


def similarity(i_d, i_col, j_d, j_col):
    tmp_i_d = []
    tmp_j_d = []
    i_mean = 0
    j_mean = 0
    for i in range(len(i_col)):
        i_c = i_col[i]
        for j in range(len(j_col)):
            j_c = j_col[j]
            if i_c == j_c:
                tmp_i_d.append(i_d[i])
                tmp_j_d.append(j_d[j])
                i_mean += i_d[i]
                j_mean += j_d[j]
                break
            elif i_c < j_c:
                break
    l = len(tmp_i_d)
    if l > 0:
        i_mean = i_mean/l
        j_mean = j_mean/l
        numerator = 0
        denom_i = 0
        denom_j = 0
        for i in range(l):
            numerator += (tmp_i_d[i]-i_mean)*(tmp_j_d[i]-j_mean)
            denom_i += (tmp_i_d[i]-i_mean)**2
            denom_j += (tmp_j_d[i]-j_mean)**2
        if denom_i == 0 or denom_j == 0:
            return 0
        return numerator/(sqrt(denom_i)*sqrt(denom_j))
    else:
        return 0


def gen_similarity_matrix(rating_csr):
    print("Start generating similarity matrix...")
    t0 = datetime.utcnow()
    uu_data = []
    uu_i = []
    uu_j = []

    for i in range(1, rating_csr.shape[0]):
        i_d = rating_csr.data[rating_csr.indptr[i]:rating_csr.indptr[i + 1]]
        i_col = rating_csr.indices[rating_csr.indptr[i]:rating_csr.indptr[i + 1]]
        for j in range(i, rating_csr.shape[0]):
            j_d = rating_csr.data[rating_csr.indptr[j]:rating_csr.indptr[j + 1]]
            j_col = rating_csr.indices[rating_csr.indptr[j]:rating_csr.indptr[j + 1]]
            sim = similarity(i_d, i_col, j_d, j_col)
            if sim > 0:
                uu_data.append(sim)
                uu_i.append(i)
                uu_j.append(j)

    similarity_matrix = csr_matrix((uu_data, (uu_i, uu_j)))
    print(datetime.utcnow() - t0)
    print("Similarity matrix complete!")
    return similarity_matrix