#!/usr/bin/env python3
from knn import UserBased


DATA = {
    "DATA_DIR": "./ml-latest-small",
    "RATING_FILE": "ratings.csv",
    "MOVIE_FILE": "movies.csv",
    "TAG_FILE": "tags.csv"
}


if __name__ == "__main__":
    user_based_cf = UserBased(DATA)
    #user_based_cf.train()
    #user_based_cf.save_sim_matrix("user_sim_matrix_small.npz")
    user_based_cf.load_sim_matrix("user_sim_matrix_small.npz")
    user_based_cf.predict_single(1, 110)
    for pred_rating, movie_id, k in user_based_cf.recommend_single(1):
        print("Predicted Rating:{}   Movie:{} {}   Actual K:{}".format(pred_rating, movie_id, user_based_cf.id_to_title(movie_id), k))
