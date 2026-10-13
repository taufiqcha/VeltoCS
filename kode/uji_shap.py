import json

import numpy as np
import shap

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri, sumbu_potensial
from model import REGRESOR

KELUARAN = HASIL / "uji_shap.json"
MODEL = "ET"
JUMLAH_TERATAS = 10
LEBAR_JENDELA_VOLT = 0.1
NAMA_CABANG = {0: "maju", 1: "balik"}


def kepentingan_shap(X, kadar):
    model = REGRESOR[MODEL](0).fit(X, np.log10(kadar))
    nilai = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    kepentingan = np.abs(nilai).mean(axis=0)
    return kepentingan / kepentingan.sum()


def kepentingan_per_jendela(kepentingan, sumbu, cabang):
    tepi = np.arange(sumbu.min(), sumbu.max() + LEBAR_JENDELA_VOLT, LEBAR_JENDELA_VOLT)
    jendela = []
    for kode, nama in NAMA_CABANG.items():
        for awal, akhir in zip(tepi, tepi[1:]):
            masuk = (cabang == kode) & (sumbu >= awal) & (sumbu < akhir)
            if masuk.any():
                jendela.append({"cabang": nama, "dari_volt": float(awal), "sampai_volt": float(akhir),
                                "porsi_shap": float(kepentingan[masuk].sum())})
    return sorted(jendela, key=lambda j: j["porsi_shap"], reverse=True)


def fitur_teratas(kepentingan, sumbu, cabang):
    urutan = np.argsort(kepentingan)[::-1][:JUMLAH_TERATAS]
    return [{"potensial_volt": float(sumbu[i]), "cabang": NAMA_CABANG[int(cabang[i])],
             "porsi_shap": float(kepentingan[i])} for i in urutan]


def main():
    kadar_icp = kadar_icp_per_label_asli()
    sumbu, cabang = sumbu_potensial()
    keluaran = {}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        kepentingan = kepentingan_shap(X, kadar)
        jendela = kepentingan_per_jendela(kepentingan, sumbu, cabang)
        keluaran[alat] = {"model": MODEL, "lebar_jendela_volt": LEBAR_JENDELA_VOLT,
                          "jendela_terpenting": jendela,
                          "fitur_teratas": fitur_teratas(kepentingan, sumbu, cabang),
                          "kepentingan_per_fitur": kepentingan.tolist()}
        for j in jendela[:5]:
            print(f"{alat} {j['cabang']:5s} {j['dari_volt']:+.2f} s.d. {j['sampai_volt']:+.2f} V  porsi SHAP {j['porsi_shap']:.4f}")
    KELUARAN.write_text(json.dumps(keluaran, indent=1))


if __name__ == "__main__":
    main()
