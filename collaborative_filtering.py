#!/usr/bin/env python3
from os import path
import pandas as pd
from scipy.sparse import csr_matrix


class CollaborativeFiltering(object):
    def __init__(self, data):
        rating = pd.read_csv(path.join(data["DATA_DIR"], data["RATING_FILE"]))
        self.rating_matrix = csr_matrix((rating.rating, (rating.userId, rating.movieId)))
        del rating
        self.movie_df = pd.read_csv(path.join(data["DATA_DIR"], data["MOVIE_FILE"]))

    def train(self):
        pass

    def predict_single(self, user_id, item_id):
        prediction = 0
        return prediction

    def recommend_single(self, user_id):
        recommendations = list()

        return recommendations

    def predict(self):
        pass

    def evaluate(self):
        pass

