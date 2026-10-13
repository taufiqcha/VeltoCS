import json

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import cohen_kappa_score

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri
from model import BENIH, KLASIFIER, REGRESOR, nomor_pita
from uji_5fold import prediksi_log_kadar, prediksi_pita
from uji_kesesuaian import bland_altman_persen, laju_salah_per_ambang, passing_bablok

KELUARAN = HASIL / "uji_label_nominal.json"


def evaluasi_terhadap_icp(X, level, kadar, target):
    rata_benih = np.mean([10 ** prediksi_log_kadar(X, level, target, REGRESOR["ET"], b) for b in BENIH], axis=0)
    benih_nol = 10 ** prediksi_log_kadar(X, level, target, REGRESOR["ET"], BENIH[0])
    tingkat = np.unique(level)
    rata = np.array([rata_benih[level == t].mean() for t in tingkat])
    acuan = np.array([kadar[level == t][0] for t in tingkat])
    recovery = 100 * rata / acuan
    pita_icp = nomor_pita(kadar)
    pita_prediksi = [prediksi_pita(X, level, nomor_pita(target), KLASIFIER["SVM"], b) for b in BENIH]
    laju = [laju_salah_per_ambang(pita_icp, p) for p in pita_prediksi]
    return {
        "recovery_median_persen": float(np.median(recovery)),
        "recovery_rentang_persen": [float(recovery.min()), float(recovery.max())],
        "t_berpasangan_p": float(ttest_rel(rata, acuan).pvalue),
        "wilcoxon_p": float(wilcoxon(rata, acuan).pvalue),
        "passing_bablok": passing_bablok(kadar, benih_nol),
        "bland_altman": bland_altman_persen(kadar, benih_nol),
        "kappa_kuadratik": float(np.mean([cohen_kappa_score(pita_icp, p, weights="quadratic") for p in pita_prediksi])),
        "per_ambang": {b: {jenis: float(np.mean([l[b][jenis] for l in laju])) for jenis in ("negatif_palsu", "positif_palsu")}
                       for b in laju[0]},
    }


def main():
    kadar_icp = kadar_icp_per_label_asli()
    keluaran = {}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        keluaran[alat] = {sumber: evaluasi_terhadap_icp(X, level, kadar, target)
                          for sumber, target in (("icp", kadar), ("nominal", level))}
        for sumber, h in keluaran[alat].items():
            print(alat, sumber, "kappa", round(h["kappa_kuadratik"], 3), "BA", [round(x, 3) for x in h["bland_altman"]["batas_kesesuaian_persen"]],
                  "recovery", round(h["recovery_median_persen"], 3), "LPP 100", round(100 * h["per_ambang"]["100"]["positif_palsu"], 3))
    KELUARAN.write_text(json.dumps(keluaran, indent=1))


if __name__ == "__main__":
    main()
