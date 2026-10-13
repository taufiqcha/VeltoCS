import json

import numpy as np
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri
from model import BATAS_PITA, BENIH, KLASIFIER, PITA, REGRESOR, nomor_pita
from uji_5fold import prediksi_log_kadar, prediksi_pita

KELUARAN = HASIL / "uji_kesesuaian.json"
REGRESOR_TERBAIK = "ET"
KLASIFIER_TERBAIK = "SVM"
TINGKAT_KEPERCAYAAN = 0.95


def passing_bablok(acuan, uji):
    i, j = np.triu_indices(len(acuan), k=1)
    beda_acuan = acuan[j] - acuan[i]
    beda_uji = uji[j] - uji[i]
    sah = beda_acuan != 0
    kemiringan = np.sort(beda_uji[sah] / beda_acuan[sah])
    kemiringan = kemiringan[kemiringan != -1]
    jumlah = len(kemiringan)
    geser = int(np.sum(kemiringan < -1))
    tengah = (jumlah - 1) / 2 + geser
    miring = float((kemiringan[int(np.floor(tengah))] + kemiringan[int(np.ceil(tengah))]) / 2)
    n = len(acuan)
    lebar = norm.ppf(1 - (1 - TINGKAT_KEPERCAYAAN) / 2) * np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    m1 = int(round((jumlah - lebar) / 2))
    bawah, atas = m1 - 1 + geser, jumlah - m1 + geser
    miring_bawah, miring_atas = float(kemiringan[bawah]), float(kemiringan[atas])
    return {
        "kemiringan": miring,
        "kemiringan_ci": [miring_bawah, miring_atas],
        "intersep": float(np.median(uji - miring * acuan)),
        "intersep_ci": [float(np.median(uji - miring_atas * acuan)),
                        float(np.median(uji - miring_bawah * acuan))],
        "jumlah_pasangan": jumlah,
    }


def bland_altman_persen(acuan, uji):
    selisih = 100 * (uji - acuan) / ((uji + acuan) / 2)
    bias, simpangan = float(np.mean(selisih)), float(np.std(selisih, ddof=1))
    bawah, atas = bias - 1.96 * simpangan, bias + 1.96 * simpangan
    return {
        "bias_persen": bias,
        "simpangan_baku_persen": simpangan,
        "batas_kesesuaian_persen": [bawah, atas],
        "porsi_di_dalam_batas": float(np.mean((selisih >= bawah) & (selisih <= atas))),
    }


def kesesuaian_kuantifikasi(X, level, kadar):
    prediksi = 10 ** prediksi_log_kadar(X, level, kadar, REGRESOR[REGRESOR_TERBAIK], BENIH[0])
    return {"model": REGRESOR_TERBAIK, "benih": BENIH[0],
            "passing_bablok": passing_bablok(kadar, prediksi),
            "bland_altman": bland_altman_persen(kadar, prediksi)}


def laju_salah_per_ambang(pita, prediksi):
    hasil = {}
    for nomor, batas in enumerate(BATAS_PITA, start=1):
        di_atas = pita >= nomor
        hasil[f"{batas:g}"] = {"negatif_palsu": float(np.mean(prediksi[di_atas] < nomor)),
                               "positif_palsu": float(np.mean(prediksi[~di_atas] >= nomor))}
    return hasil


def metrik_ordinal(pita, prediksi):
    matriks = confusion_matrix(pita, prediksi, labels=range(len(PITA)))
    salah = pita != prediksi
    jarak = np.abs(pita - prediksi)
    per_pita = {}
    for indeks, nama in enumerate(PITA):
        benar_positif = matriks[indeks, indeks]
        positif = matriks[indeks].sum()
        negatif_benar = matriks.sum() - positif - matriks[:, indeks].sum() + benar_positif
        per_pita[nama] = {"sensitivitas": benar_positif / positif,
                          "spesifisitas": negatif_benar / (matriks.sum() - positif)}
    return {
        "kappa_kuadratik": cohen_kappa_score(pita, prediksi, weights="quadratic"),
        "kappa_linear": cohen_kappa_score(pita, prediksi, weights="linear"),
        "porsi_salah_ke_pita_tetangga": float(np.mean(jarak[salah] == 1)) if salah.any() else 1.0,
        "jumlah_salah_lebih_satu_pita": int(np.sum(jarak > 1)),
        "per_pita": per_pita,
        "per_ambang": laju_salah_per_ambang(pita, prediksi),
    }


def rerata_bertingkat(daftar):
    contoh = daftar[0]
    if isinstance(contoh, dict):
        return {kunci: rerata_bertingkat([d[kunci] for d in daftar]) for kunci in contoh}
    return [float(np.mean(daftar)), float(np.std(daftar))]


def kesesuaian_semi_kuantifikasi(X, level, kadar):
    pita = nomor_pita(kadar)
    per_benih = [metrik_ordinal(pita, prediksi_pita(X, level, pita, KLASIFIER[KLASIFIER_TERBAIK], benih))
                 for benih in BENIH]
    return {"model": KLASIFIER_TERBAIK, **rerata_bertingkat(per_benih)}


def cetak_ringkasan(alat, hasil):
    pb, ba, ordinal = hasil["passing_bablok"], hasil["bland_altman"], hasil["semi_kuantifikasi"]
    print(f"{alat} Passing-Bablok kemiringan {pb['kemiringan']:.4f} CI {pb['kemiringan_ci']}"
          f" intersep {pb['intersep']:.4f} CI {pb['intersep_ci']}")
    print(f"{alat} Bland-Altman bias {ba['bias_persen']:.4f}% batas {ba['batas_kesesuaian_persen']}"
          f" di dalam batas {ba['porsi_di_dalam_batas']:.4f}")
    print(f"{alat} kappa kuadratik {ordinal['kappa_kuadratik'][0]:.4f} linear {ordinal['kappa_linear'][0]:.4f}"
          f" salah ke tetangga {ordinal['porsi_salah_ke_pita_tetangga'][0]:.4f}"
          f" salah lebih satu pita {ordinal['jumlah_salah_lebih_satu_pita'][0]:.1f}")
    for batas, laju in ordinal["per_ambang"].items():
        print(f"{alat} ambang {batas} mg/L negatif palsu {100 * laju['negatif_palsu'][0]:.3f}%"
              f" positif palsu {100 * laju['positif_palsu'][0]:.3f}%")


def main():
    kadar_icp = kadar_icp_per_label_asli()
    keluaran = {}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        hasil = {**kesesuaian_kuantifikasi(X, level, kadar),
                 "semi_kuantifikasi": kesesuaian_semi_kuantifikasi(X, level, kadar)}
        cetak_ringkasan(alat, hasil)
        keluaran[alat] = hasil
    KELUARAN.write_text(json.dumps(keluaran, indent=1, default=float))


if __name__ == "__main__":
    main()
