import joblib
import pandas as pd
import warnings

df = pd.read_csv("C:/Users/mehmet kupeli/PycharmProjects/pythonProject/datasets/data.csv", engine='python', sep=None)

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

from Student_Pipeline_v2 import students_data_prep

X, y = students_data_prep(df)

random_user = X.sample(1, random_state=250)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
