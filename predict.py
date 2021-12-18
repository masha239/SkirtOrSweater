import pickle

from utils import *


def predict_image(filename: str) -> str:
    picture = prepare_picture1(filename, n=400)
    df = make_features([picture], 20, 100)
    with open('model.pkl', 'rb') as file:
        rf = pickle.load(file)
    return rf.predict(df)[0]
