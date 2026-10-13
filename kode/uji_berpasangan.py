import json

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri
from model import BENIH, REGRESOR
from uji_5fold import prediksi_log_kadar

KELUARAN = HASIL / "uji_berpasangan.json"
MODEL = "ET"


def prediksi_rata_benih(X, level, kadar):
    return np.mean([10 ** prediksi_log_kadar(X, level, kadar, REGRESOR[MODEL], benih) for benih in BENIH], axis=0)


def rerata_per_level(level, kadar, prediksi):
    hasil = {}
    for nilai in np.unique(level):
        pilih = level == nilai
        rata = float(np.mean(prediksi[pilih]))
        hasil[float(nilai)] = {"kadar_icp": float(kadar[pilih][0]), "rata_prediksi": rata,
                               "bias_persen": float(100 * (rata - kadar[pilih][0]) / kadar[pilih][0])}
    return hasil


def uji_berpasangan(per_level):
    icp = np.array([v["kadar_icp"] for v in per_level.values()])
    prediksi = np.array([v["rata_prediksi"] for v in per_level.values()])
    recovery = 100 * prediksi / icp
    return {"t_berpasangan_p": float(ttest_rel(prediksi, icp).pvalue),
            "t_berpasangan_log_p": float(ttest_rel(np.log10(prediksi), np.log10(icp)).pvalue),
            "wilcoxon_p": float(wilcoxon(prediksi, icp).pvalue),
            "recovery_median_persen": float(np.median(recovery))}


def selisih_antar_alat(per_alat):
    hasil = {}
    for nilai in per_alat[ALAT[0]]:
        a, b = (per_alat[alat][nilai]["rata_prediksi"] for alat in ALAT)
        hasil[nilai] = float(100 * abs(a - b) / per_alat[ALAT[0]][nilai]["kadar_icp"])
    return {"selisih_persen_per_level": hasil, "selisih_median_persen": float(np.median(list(hasil.values())))}


def main():
    kadar_icp = kadar_icp_per_label_asli()
    keluaran, per_alat = {"model": MODEL}, {}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        per_level = rerata_per_level(level, kadar, prediksi_rata_benih(X, level, kadar))
        per_alat[alat] = per_level
        keluaran[alat] = {"per_level": per_level, "uji_berpasangan": uji_berpasangan(per_level)}
        print(alat, {n: round(v, 3) for n, v in keluaran[alat]["uji_berpasangan"].items()})
    keluaran["antar_alat"] = selisih_antar_alat(per_alat)
    print("selisih LecSens-EmStat4 median", round(keluaran["antar_alat"]["selisih_median_persen"], 3), "%")
    KELUARAN.write_text(json.dumps(keluaran, indent=1))


if __name__ == "__main__":
    main()
