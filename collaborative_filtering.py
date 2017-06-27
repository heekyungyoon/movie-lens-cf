#!/usr/bin/env python3
from os import path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from numpy import sqrt
from datetime import datetime

DATA_DIR = "./ml-latest-small"
RATING_FILE = "ratings.csv"
MOVIE_FILE = "movies.csv"
TAG_FILE = "tags.csv"


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


class CollaborativeFiltering(object):
    def __init__(self):
        self.rating_matrix = None
        self.sim_matrix = None
        self.movie_df = pd.read_csv(path.join(DATA_DIR, MOVIE_FILE))

    def train(self, train_set):
        pass

    def predict_single(self, user_id, movie_id):
        prediction = 0
        return prediction

    def recommend_single(self, user_id):
        recommendations = list()

        return recommendations

    def predict(self):
        pass

    def evaluate(self):
        pass


class UserBased(CollaborativeFiltering):
    def __init__(self, train_set):
        CollaborativeFiltering.__init__(self)
        self.rating_matrix = train_set

    def train(self):
        if self.sim_matrix is None:
            self.sim_matrix = gen_similarity_matrix(self.train_set)

    def save_sim_matrix(self, filename):
        print("Start saving similarity matrix...")
        t0 = datetime.utcnow()
        np.savez(filename,
                 data=self.sim_matrix.data, indices=self.sim_matrix.indices, indptr=self.sim_matrix.indptr,
                 shape=self.sim_matrix.shape)
        print(datetime.utcnow() - t0)
        print("Saved!")
        return

    def load_sim_matrix(self, filename):
        loader = np.load(filename)
        self.sim_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    def predict_single(self, user_id, movie_id, top_k=30):
        items_to_exclude = self.rating_matrix.indices[
                           self.rating_matrix.indptr[user_id]:self.rating_matrix.indptr[user_id + 1]]
        if movie_id in items_to_exclude:
            raise ValueError

        user_similarities = list()
        for i in range(self.sim_matrix.indptr[user_id],self.sim_matrix.indptr[user_id+1]):
            s = self.sim_matrix.data[i]
            if s > 0 and self.sim_matrix.indices[i] != user_id:
                user_similarities.append((self.sim_matrix.data[i], self.sim_matrix.indices[i]))
        user_similarities = sorted(user_similarities, reverse=True)

        numerator = 0
        denominator = 0
        counter = 0
        for s, u in user_similarities[:top_k*2]:
            rated_items = self.rating_matrix.indices[self.rating_matrix.indptr[u]:self.rating_matrix.indptr[u+1]]
            for i in range(len(rated_items)):
                r_item = rated_items[i]
                if r_item == movie_id:
                    idx = i
                    rating = self.rating_matrix.data[self.rating_matrix.indptr[u]+idx]
                    numerator += s * rating
                    denominator += s
                    counter += 1
                    break
                elif r_item > movie_id:
                    break
            if counter == top_k:
                break

        if denominator <= 0:
            return 0, counter
        else:
            return numerator/denominator, counter

    def recommend_single(self, user_id, top_k=30):
        recommendations = list()
        similar_users = list()
        for i in range(self.sim_matrix.indptr[user_id], self.sim_matrix.indptr[user_id + 1]):
            s = self.sim_matrix.data[i]
            if s > 0 and self.sim_matrix.indices[i] != user_id:
                similar_users.append((self.sim_matrix.data[i], self.sim_matrix.indices[i]))
        similar_users = sorted(similar_users, reverse=True)

        recommend_dict = dict()
        items_to_exclude = self.rating_matrix.indices[self.rating_matrix.indptr[user_id]:self.rating_matrix.indptr[user_id + 1]]
        for sim, sim_user_id in similar_users[:top_k]:
            for i in range(self.rating_matrix.indptr[sim_user_id], self.rating_matrix.indptr[sim_user_id + 1]):
                movie_id = self.rating_matrix.indices[i]
                if movie_id not in items_to_exclude:
                    rating = self.rating_matrix.data[i]
                    if recommend_dict.get(movie_id):
                        recommend_dict[movie_id].append((sim, rating))
                    else:
                        recommend_dict[movie_id] = [(sim, rating)]

        for movie_id in recommend_dict:
            numerator = 0
            denominator = 0
            for sim, rating in recommend_dict[movie_id]:
                numerator += sim * rating
                denominator += sim
            if denominator > 0:
                recommendations.append((numerator/denominator, movie_id, len(recommend_dict[movie_id])))
        recommendations = sorted(recommendations, reverse=True)
        return recommendations[:top_k]

    def id_to_title(self, movie_id):
        return list(self.movie_df.loc[self.movie_df.movieId == movie_id, "title"])[0]

    def predict(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    rating = pd.read_csv(path.join(DATA_DIR, RATING_FILE))
    rating_csr = csr_matrix((rating.rating, (rating.userId, rating.movieId)))
    user_based_cf = UserBased(rating_csr)
    #user_based_cf.train()
    #user_based_cf.save_sim_matrix("user_sim_matrix_small.npz")
    user_based_cf.load_sim_matrix("user_sim_matrix_small.npz")
    user_based_cf.predict_single(1, 110)
    for pred_rating, movie_id, k in user_based_cf.recommend_single(1):
        print("Predicted Rating:{}   Movie:{} {}   Actual K:{}".format(pred_rating, movie_id, user_based_cf.id_to_title(movie_id), k))
