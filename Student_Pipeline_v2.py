################################################
# Helper Functions
################################################

# utils.py
# helpers.py

# Data Preprocessing & Feature Engineering

################################################
# End-to-End Student_Fate  Machine Learning Pipeline
################################################

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


################################################
# Helper Functions
################################################

# utils.py
# helpers.py

# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=15, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def students_data_prep(df):
    df['Target'] = df['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df = df.reset_index(drop=False)
    # Verisetine StudenetID eklendi
    Student_ID = 'Student_ID'
    df.rename(columns={"index": Student_ID}, inplace=True)
    df["Student_ID"] = df["Student_ID"] + 1
    # Değişken isimleri büyütmek ve boşluk varsa ortadan kaldırmak
    df.columns = [col.upper().replace(" ", "_") for col in df.columns]
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15, car_th=30)
    cat_cols = [col for col in cat_cols if "TARGET" not in col]
    # Threshld üstünde kalan ama kategorik olan değişkenleri manuel ekledim
    deltas = ['APPLICATION_MODE', 'PREVIOUS_QUALIFICATION', 'NACIONALITY', "MOTHER'S_QUALIFICATION",
              "FATHER'S_QUALIFICATION", "MOTHER'S_OCCUPATION", "FATHER'S_OCCUPATION", "COURSE"]
    cat_cols += [deltas for deltas in deltas]
    num_cols = list(set(num_cols) - set(deltas))

    df = one_hot_encoder(df, cat_cols, drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15, car_th=30)

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["TARGET"]
    X = df.drop(["TARGET", "STUDENT_ID"], axis=1)

    return X, y


# Base Models
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVM", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# Hyperparameter Optimization

# config.py

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.7, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

SVM_params = {"C": [0.1, 1, 10],  # Ceza parametresi
              "kernel": ["linear", "rbf", "sigmoid"]}  # Çekirdek türü

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss'), xgboost_params),
               ('SVM', SVC(), SVM_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"]),
                                              ('SVM', best_models["SVM"])], voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


"""

GBM (Gradient Boosting Machine): 0.9262
LR (Logistic Regression): 0.9255
SVC (Support Vector Classifier): 0.9243
LightGBM: 0.9208
Adaboost: 0.9199
RF (Random Forest): 0.9196
XGBoost: 0.9149
KNN (K-Nearest Neighbors): 0.8435
CART (Decision Tree): 0.7769 """


################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv("C:/Users/mehmet kupeli/PycharmProjects/pythonProject/datasets/data.csv", engine='python',
                     sep=None)
    X, y = students_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf


if __name__ == "__main__":
    print("İşlem başladı")
    main()

# git github
# makefile
# veri tabanlarından
# log
# class
# docker
# requirement.txt

# Çıktı
"""
roc_auc: 0.9255 (LR) 
roc_auc: 0.8435 (KNN) 
roc_auc: 0.9243 (SVC) 
roc_auc: 0.7769 (CART) 
roc_auc: 0.9196 (RF) 
roc_auc: 0.9199 (Adaboost) 
roc_auc: 0.9262 (GBM)
roc_auc: 0.9149 (XGBoost) 
roc_auc: 0.9208 (LightGBM) """

"""Hyperparameter Optimization....
########## KNN ##########
roc_auc (Before): 0.8435
roc_auc (After): 0.8747
KNN best params: {'n_neighbors': 14}

########## CART ##########
roc_auc (Before): 0.7758
roc_auc (After): 0.8831
CART best params: {'max_depth': 4, 'min_samples_split': 27}

########## RF ##########
roc_auc (Before): 0.9191
roc_auc (After): 0.9132
RF best params: {'max_depth': None, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 100}

########## XGBoost ##########
roc_auc (Before): 0.9149
roc_auc (After): 0.925
XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 200}

########## LightGBM ##########
roc_auc (Before): 0.9208
roc_auc (After): 0.9268
LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 300}
"""
# 3 model vardı : KNN; LightGBM; RF ile sonuçlar cv=3
# Accuracy: 0.8453884659905403
# F1Score: 0.8546316749524577
# ROC_AUC: 0.9205037180877262

# 4 model vardı : KNN; LightGBM; RF; SVM ile sonuçlar cv=3
"""
Accuracy: 0.8471966822283038
F1Score: 0.8558575882084867
ROC_AUC: 0.9210272342474344

"""
# Voting için model seçimi;
"""Boosting Temelli Modeller:

1.GBM (Gradient Boosting Machine): 0.9262
2. Adaboost: 0.9199
3. XGBoost: 0.9149
Sınıflandırma ve Lineer Modeller:
4. LR (Logistic Regression): 0.9255

Kernel Temelli Modeller:
5. SVC (Support Vector Classifier): 0.9243

Ensemble Modeller:
6. RF (Random Forest): 0.9196

Bu gruplamalar, modellerin temel yaklaşımlarına dayanarak sınıflandırılmıştır. 
Boosting temelli modeller, ağırlıklı olarak boosting yöntemlerini kullanırken, 
sınıflandırma ve lineer modeller sınıflandırma ve lineer regresyon temelli yöntemleri içerir. 
Kernel temelli modeller,kernel fonksiyonlarını kullanır ve ensemble modeller birçok alt modelin 
birleşimini içerir.

"""
