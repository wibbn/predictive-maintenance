import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import create_gbm_dataset

def train_gbm(n_epochs=100):
    df = pd.read_csv('./data/df_super.csv')

    x, y = create_gbm_dataset(df)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.9)

    model = CatBoostClassifier(
        iterations=n_epochs, 
        learning_rate=0.01
        )

    model.fit(xtrain, ytrain, eval_set=(xtest, ytest))

    model.save_model('./models/gbm.cbm')

if __name__ == "__main__":
    train_gbm()