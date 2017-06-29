#!/usr/bin/env python3
from scipy.sparse import save_npz, load_npz
from collaborative_filtering import CollaborativeFiltering
from similarity import *


class UserBased(CollaborativeFiltering):
    def __init__(self, data):
        CollaborativeFiltering.__init__(self, data)
        self.sim_matrix = None

    def train(self):
        if self.sim_matrix is None:
            self.sim_matrix = gen_similarity_matrix(self.train_set)

    def save_sim_matrix(self, filename):
        print("Start saving similarity matrix...")
        t0 = datetime.utcnow()
        save_npz(filename, self.sim_matrix)
        print(datetime.utcnow() - t0)
        print("Saved!")
        return

    def load_sim_matrix(self, filename):
        self.sim_matrix = load_npz(filename)

    def predict_single(self, user_id, item_id, top_k=30):
        items_to_exclude = self.rating_matrix.indices[
                           self.rating_matrix.indptr[user_id]:self.rating_matrix.indptr[user_id + 1]]
        if item_id in items_to_exclude:
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
                if r_item == item_id:
                    idx = i
                    rating = self.rating_matrix.data[self.rating_matrix.indptr[u]+idx]
                    numerator += s * rating
                    denominator += s
                    counter += 1
                    break
                elif r_item > item_id:
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
                item_id = self.rating_matrix.indices[i]
                if item_id not in items_to_exclude:
                    rating = self.rating_matrix.data[i]
                    if recommend_dict.get(item_id):
                        recommend_dict[item_id].append((sim, rating))
                    else:
                        recommend_dict[item_id] = [(sim, rating)]

        for item_id in recommend_dict:
            numerator = 0
            denominator = 0
            for sim, rating in recommend_dict[item_id]:
                numerator += sim * rating
                denominator += sim
            if denominator > 0:
                recommendations.append((numerator/denominator, item_id, len(recommend_dict[item_id])))
        recommendations = sorted(recommendations, reverse=True)
        return recommendations[:top_k]

    def id_to_title(self, item_id):
        return list(self.movie_df.loc[self.movie_df.movieId == item_id, "title"])[0]

    def predict(self):
        pass

    def evaluate(self):
        pass