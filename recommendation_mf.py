#!/usr/bin/env python3
from matrix_factorization import MatrixFactorization


DATA = {
    "DATA_DIR": "./ml-latest-small",
    "RATING_FILE": "ratings.csv",
    "MOVIE_FILE": "movies.csv",
    "TAG_FILE": "tags.csv"
}


if __name__ == "__main__":
    mf_cf = MatrixFactorization(DATA)
    mf_cf.train()
    mf_cf.predict_single(1, 110)
