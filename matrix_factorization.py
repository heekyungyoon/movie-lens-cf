#!/usr/bin/env python3
import numpy as np
from scipy.sparse import save_npz, load_npz
from collaborative_filtering import CollaborativeFiltering


class MatrixFactorization(CollaborativeFiltering):
    def __init__(self, data, k=4, alpha=0.02, lambda_user=0.02, lambda_item=0.02, n_iter=5):
        CollaborativeFiltering.__init__(self, data)
        self.k = k
        np.random.seed(323)
        self.user_factor = 0.1 * np.random.rand(self.rating_matrix.shape[0], self.k)
        self.item_factor = 0.1 * np.random.rand(self.rating_matrix.shape[1], self.k)
        self.alpha = alpha
        self.lambda_u = lambda_user
        self.lambda_i = lambda_item
        self.n_iter = n_iter

    def _update_weights(self, user_id, item_id, diff):
        for k_idx in range(self.k):
            self.user_factor[user_id, k_idx] = max(0.0, self.user_factor[user_id, k_idx] + self.alpha * (diff * self.item_factor[item_id, k_idx] - self.lambda_u * self.user_factor[user_id, k_idx]))
            self.item_factor[item_id, k_idx] = max(0.0, self.item_factor[item_id, k_idx] + self.alpha * (diff * self.user_factor[user_id, k_idx] - self.lambda_i * self.item_factor[item_id, k_idx]))
        return

    def _rmse(self):
        error = 0
        for user_id in range(self.rating_matrix.shape[0]):
            for i_idx in range(self.rating_matrix.indptr[user_id], self.rating_matrix.indptr[user_id + 1]):
                item_id = self.rating_matrix.indices[i_idx]
                rating = self.rating_matrix.data[i_idx]
                error += (rating - np.dot(self.user_factor[user_id], self.item_factor[item_id]))**2
        return np.sqrt(error/len(self.rating_matrix.data))

    def train(self):
        for it in range(self.n_iter):
            for user_id in range(self.rating_matrix.shape[0]):
                for i_idx in range(self.rating_matrix.indptr[user_id],self.rating_matrix.indptr[user_id+1]):
                    item_id = self.rating_matrix.indices[i_idx]
                    rating = self.rating_matrix.data[i_idx]
                    diff = rating - np.dot(self.user_factor[user_id], self.item_factor[item_id])
                    self._update_weights(user_id, item_id, diff)
            error = self._rmse()
            print("iter: {} error: {}".format(it, error))

    def predict_single(self, user_id, movie_id):
        prediction = 0
        return prediction

    def recommend_single(self, user_id):
        recommendations = list()

        return recommendations

    def predict(self):
        pass