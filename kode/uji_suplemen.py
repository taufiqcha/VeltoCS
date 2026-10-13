import json

import numpy as np
from scipy.stats import spearmanr

from data import ALAT, HASIL, SCAN, kadar_icp_per_label_asli, muat_voltametri
from model import BENIH, KLASIFIER, PITA, nomor_pita
from uji_5fold import prediksi_pita

KELUARAN = HASIL / "uji_suplemen.json"
KISI = np.round(np.arange(-1.40, -0.50 + 0.01 / 2, 0.01), 4)
BUANG_SCAN = 2
MIN_TITIK = 15
LEVEL_CONTOH = 100.0


def satu_arah(v, i, naik):
    arah = np.sign(np.diff(v))
    pilih = arah > 0 if naik else arah < 0
    if pilih.sum() < MIN_TITIK:
        return np.full(KISI.size, np.nan)
    indeks = np.zeros(v.size, dtype=bool)
    indeks[:-1] |= pilih
    indeks[1:] |= pilih
    vv, ii = v[indeks], i[indeks]
    urut = np.argsort(vv)
    vv, ii = vv[urut], ii[urut]
    vv, awal = np.unique(np.round(vv, 6), return_inverse=True)
    ii = np.bincount(awal, weights=ii) / np.bincount(awal)
    return np.interp(KISI, vv, ii, left=np.nan, right=np.nan)


def selisih_per_scan(V, I, ok):
    kurva = np.array([np.concatenate([satu_arah(V[s], I[s], False), satu_arah(V[s], I[s], True)]) if ok[s]
                      else np.full(2 * KISI.size, np.nan) for s in range(V.shape[0])])
    with np.errstate(all="ignore"):
        acuan = np.nanmedian(kurva[BUANG_SCAN:], axis=0)
        return np.nanmedian(np.abs(kurva - acuan), axis=1) / np.nanmedian(np.abs(acuan))


def kestabilan_scan(alat):
    arsip = np.load(SCAN / f"scans_{alat}.npz")
    V, I, y, ok = arsip["V"], arsip["I"], arsip["y"].astype(float), arsip["ok"]
    selisih = np.array([selisih_per_scan(V[n], I[n], ok[n]) for n in range(V.shape[0])])
    return {"selisih_median_persen": (100 * np.nanmedian(selisih, axis=0)).tolist(),
            "jumlah_pengukuran": int(V.shape[0]), "contoh": int(np.flatnonzero(y == LEVEL_CONTOH)[0]), "level_contoh": LEVEL_CONTOH}


def korelasi_per_potensial(X, kadar):
    return [float(spearmanr(X[:, j], kadar)[0]) for j in range(X.shape[1])]


def matriks_label_nominal(X, level, kadar):
    pita_icp = nomor_pita(kadar)
    matriks = np.zeros((len(PITA), len(PITA)), dtype=int)
    for benih in BENIH:
        np.add.at(matriks, (pita_icp, prediksi_pita(X, level, nomor_pita(level), KLASIFIER["SVM"], benih)), 1)
    return matriks.tolist()


def main():
    kadar_icp = kadar_icp_per_label_asli()
    keluaran = {}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        keluaran[alat] = {"scan": kestabilan_scan(alat), "spearman_per_potensial": korelasi_per_potensial(X, kadar),
                          "matriks_label_nominal": matriks_label_nominal(X, level, kadar)}
        print(alat, "selisih per scan (%)", np.round(keluaran[alat]["scan"]["selisih_median_persen"], 1),
              "| matriks nominal", keluaran[alat]["matriks_label_nominal"])
    KELUARAN.write_text(json.dumps(keluaran, indent=1))


if __name__ == "__main__":
    main()
