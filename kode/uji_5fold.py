import json

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri
from model import BENIH, KLASIFIER, PITA, REGRESOR, lipatan, nomor_pita

KELUARAN = HASIL / "uji_5fold.json"


def prediksi_log_kadar(X, strata, kadar, buat_model, benih):
    log_kadar = np.log10(kadar)
    prediksi = np.empty(len(kadar))
    for latih, uji in lipatan(strata, benih):
        model = buat_model(benih).fit(X[latih], log_kadar[latih])
        batas_bawah, batas_atas = log_kadar[latih].min(), log_kadar[latih].max()
        prediksi[uji] = np.clip(np.ravel(model.predict(X[uji])), batas_bawah, batas_atas)
    return prediksi


def metrik_kuantifikasi(kadar, log_prediksi):
    prediksi = 10 ** log_prediksi
    return {
        "r2": r2_score(kadar, prediksi),
        "r2_log": r2_score(np.log10(kadar), log_prediksi),
        "rmse": mean_squared_error(kadar, prediksi) ** 0.5,
        "mae": mean_absolute_error(kadar, prediksi),
    }


def prediksi_pita(X, strata, pita, buat_model, benih):
    prediksi = np.empty(len(pita), dtype=int)
    for latih, uji in lipatan(strata, benih):
        prediksi[uji] = buat_model(benih).fit(X[latih], pita[latih]).predict(X[uji])
    return prediksi


def metrik_semi_kuantifikasi(pita, prediksi):
    return {
        "akurasi": accuracy_score(pita, prediksi),
        "f1_makro": f1_score(pita, prediksi, average="macro"),
        "matriks": confusion_matrix(pita, prediksi, labels=range(len(PITA))).tolist(),
    }


def rerata_benih(daftar_metrik):
    angka = [nama for nama in daftar_metrik[0] if nama != "matriks"]
    return {nama: [float(np.mean([m[nama] for m in daftar_metrik])),
                   float(np.std([m[nama] for m in daftar_metrik]))] for nama in angka}


def uji_kuantifikasi(X, level, kadar):
    hasil = {}
    for nama in REGRESOR:
        hasil[nama] = {
            label: rerata_benih([metrik_kuantifikasi(target, prediksi_log_kadar(X, level, target, REGRESOR[nama], benih))
                                 for benih in BENIH])
            for label, target in (("icp", kadar), ("timbangan", level))
        }
    return hasil


def uji_semi_kuantifikasi(X, level, kadar):
    pita = nomor_pita(kadar)
    hasil = {}
    for nama in KLASIFIER:
        per_benih = [metrik_semi_kuantifikasi(pita, prediksi_pita(X, level, pita, KLASIFIER[nama], benih))
                     for benih in BENIH]
        hasil[nama] = {**rerata_benih(per_benih), "matriks_benih_0": per_benih[0]["matriks"]}
    return hasil, np.bincount(pita, minlength=len(PITA)).tolist()


def cetak_ringkasan(alat, hasil):
    for nama, label in hasil["kuantifikasi"].items():
        print(f"{alat} {nama:4s} R2 icp {label['icp']['r2'][0]:.4f}  R2 log icp {label['icp']['r2_log'][0]:.4f}"
              f"  |  R2 timbangan {label['timbangan']['r2'][0]:.4f}  R2 log timbangan {label['timbangan']['r2_log'][0]:.4f}")
    for nama, metrik in hasil["semi_kuantifikasi"].items():
        print(f"{alat} {nama:4s} akurasi pita {metrik['akurasi'][0]:.4f}  F1 makro {metrik['f1_makro'][0]:.4f}")


def main():
    kadar_icp = kadar_icp_per_label_asli()
    keluaran = {"pita": dict(zip(PITA, ["< 10", "10-100", "100-1000", ">= 1000"])), "alat": {}}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        semi, jumlah_per_pita = uji_semi_kuantifikasi(X, level, kadar)
        hasil = {"jumlah_sapuan": len(level), "jumlah_per_pita": jumlah_per_pita,
                 "kuantifikasi": uji_kuantifikasi(X, level, kadar), "semi_kuantifikasi": semi}
        cetak_ringkasan(alat, hasil)
        keluaran["alat"][alat] = hasil
    KELUARAN.write_text(json.dumps(keluaran, indent=1))


if __name__ == "__main__":
    main()
