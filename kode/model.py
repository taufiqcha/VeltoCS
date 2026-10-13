import numpy as np
from lightgbm import LGBMClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor

JUMLAH_LIPATAN = 5
BENIH = range(5)
PITA = ("rendah", "sedang", "tinggi", "sangat tinggi")
BATAS_PITA = (10, 100, 1000)

REGRESOR = {
    "RF": lambda benih: RandomForestRegressor(300, n_jobs=-1, random_state=benih),
    "ET": lambda benih: ExtraTreesRegressor(300, n_jobs=-1, random_state=benih),
    "XGB": lambda benih: XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                      subsample=0.8, n_jobs=-1, random_state=benih),
    "SVR": lambda benih: make_pipeline(StandardScaler(), SVR(C=10, gamma="scale")),
    "PLS": lambda benih: PLSRegression(10, scale=True),
}

KLASIFIER = {
    "RF": lambda benih: RandomForestClassifier(300, n_jobs=-1, random_state=benih),
    "ET": lambda benih: ExtraTreesClassifier(300, n_jobs=-1, random_state=benih),
    "LGBM": lambda benih: LGBMClassifier(n_estimators=300, random_state=benih, verbose=-1),
    "SVM": lambda benih: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
}


def nomor_pita(kadar):
    return np.searchsorted(BATAS_PITA, kadar, side="right")


def lipatan(strata, benih):
    pembagi = StratifiedKFold(JUMLAH_LIPATAN, shuffle=True, random_state=benih)
    return pembagi.split(np.zeros(len(strata)), strata)
