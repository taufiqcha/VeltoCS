from pathlib import Path

import numpy as np
import pandas as pd

AKAR = Path(__file__).resolve().parent.parent
DATA = AKAR / "data"
HASIL = AKAR / "results"
ICP_OES = DATA / "icp_oes.csv"
VOLTAMETRI = DATA / "sumbu_sama.npz"
SCAN = DATA
ALAT = ("LC", "EM")


def kadar_icp_per_label_asli():
    tabel = pd.read_csv(ICP_OES)
    rata = tabel[["replicate_1_mg_L", "replicate_2_mg_L"]].mean(axis=1)
    return {float(nominal): float(nilai) for nominal, nilai in zip(tabel["nominal_mg_L"], rata)}


def muat_voltametri(alat, kadar_icp):
    arsip = np.load(VOLTAMETRI)
    X = arsip[f"X_{alat}"]
    level = arsip[f"y_{alat}"].astype(float)
    punya_icp = np.isin(level, list(kadar_icp))
    level = level[punya_icp]
    return X[punya_icp], level, np.array([kadar_icp[nilai] for nilai in level])


def sumbu_potensial():
    arsip = np.load(VOLTAMETRI)
    return arsip["sumbu"], arsip["cabang"]
