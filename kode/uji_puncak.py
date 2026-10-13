import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri, sumbu_potensial

matplotlib.use("Agg")

KELUARAN = HASIL / "uji_puncak.json"
GAMBAR = HASIL / "uji_puncak.png"
SHAP = HASIL / "uji_shap.json"
NAMA_CABANG = {0: "maju", 1: "balik"}
BENTUK_PUNCAK = {"lembah": -1, "bukit": 1}
PROMINENSI_MINIMUM = 3.0
RADIUS_PELACAKAN_VOLT = 0.05
JUMLAH_JENDELA_SHAP = 2
NAMA_ALAT = {"LC": "LecSens", "EM": "EmStat4"}


def rata_per_level(X, level, indeks):
    return {float(nilai): X[level == nilai][:, indeks].mean(axis=0) for nilai in np.unique(level)}


def puncak_terkuat_per_level(rata, potensial, tanda):
    temuan = []
    for nilai, kurva in sorted(rata.items()):
        posisi, sifat = find_peaks(tanda * kurva, prominence=PROMINENSI_MINIMUM)
        if len(posisi):
            terkuat = int(np.argmax(sifat["prominences"]))
            temuan.append({"level": nilai, "potensial_volt": float(potensial[posisi[terkuat]]),
                           "prominensi": float(sifat["prominences"][terkuat])})
    return temuan


def puncak_cabang(rata, potensial):
    calon = {bentuk: puncak_terkuat_per_level(rata, potensial, tanda) for bentuk, tanda in BENTUK_PUNCAK.items()}
    bentuk = max(calon, key=lambda b: sum(t["prominensi"] for t in calon[b]))
    letak = [t["potensial_volt"] for t in calon[bentuk]]
    return {"bentuk": bentuk, "per_level": calon[bentuk], "jumlah_level_berpuncak": len(letak),
            "jumlah_level": len(rata), "median_volt": float(np.median(letak)),
            "rentang_volt": [float(min(letak)), float(max(letak))]}


def korelasi_tinggi_puncak(rata, potensial, puncak, kadar_level):
    dekat = np.abs(potensial - puncak["median_volt"]) <= RADIUS_PELACAKAN_VOLT
    tanda = BENTUK_PUNCAK[puncak["bentuk"]]
    tinggi = [float(np.max(tanda * rata[nilai][dekat])) for nilai in sorted(rata)]
    return float(spearmanr(kadar_level, tinggi)[0])


def korelasi_terkuat(X, kadar, indeks, potensial):
    rho = np.array([spearmanr(X[:, j], kadar)[0] for j in indeks])
    terbesar = int(np.argmax(np.abs(rho)))
    return {"potensial_volt": float(potensial[terbesar]), "spearman": float(rho[terbesar])}


def analisis_cabang(X, level, kadar, indeks, potensial):
    rata = rata_per_level(X, level, indeks)
    kadar_level = [kadar[level == nilai][0] for nilai in sorted(rata)]
    puncak = puncak_cabang(rata, potensial)
    puncak["spearman_tinggi_lawan_kadar_icp"] = korelasi_tinggi_puncak(rata, potensial, puncak, kadar_level)
    return {"puncak": puncak, "korelasi_terkuat": korelasi_terkuat(X, kadar, indeks, potensial)}, rata


def gambar_panel(ax, potensial, rata, puncak, jendela_shap):
    peta = plt.cm.viridis(np.linspace(0, 1, len(rata)))
    for warna, (nilai, kurva) in zip(peta, sorted(rata.items())):
        ax.plot(potensial, kurva, color=warna, lw=1, label=f"{nilai:g}")
    for j in jendela_shap:
        ax.axvspan(j["dari_volt"], j["sampai_volt"], color="tab:red", alpha=0.15)
    ax.axvspan(*puncak["rentang_volt"], color="tab:blue", alpha=0.2)
    ax.axvline(puncak["median_volt"], color="tab:blue", ls="--", lw=1)


def gambar(hasil, rata_semua, potensial_semua, shap):
    kanvas, panel = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for baris, alat in enumerate(ALAT):
        for kolom, nama in enumerate(NAMA_CABANG.values()):
            jendela = [j for j in shap[alat]["jendela_terpenting"] if j["cabang"] == nama][:JUMLAH_JENDELA_SHAP]
            gambar_panel(panel[baris, kolom], potensial_semua[nama], rata_semua[alat][nama],
                         hasil[alat][nama]["puncak"], jendela)
            panel[baris, kolom].set_title(f"{NAMA_ALAT[alat]}, cabang {nama}")
            panel[baris, kolom].set_ylabel("arus")
    for ax in panel[1]:
        ax.set_xlabel("potensial (V)")
    panel[0, 1].legend(title="label (ppm)", fontsize=7, ncol=2)
    penanda = [Patch(color="tab:blue", alpha=0.2, label="wilayah puncak (rentang antar level)"),
               Line2D([], [], color="tab:blue", ls="--", lw=1, label="median letak puncak"),
               Patch(color="tab:red", alpha=0.15, label="jendela SHAP terpenting")]
    kanvas.legend(handles=penanda, loc="lower center", ncol=3, fontsize=9)
    kanvas.tight_layout(rect=(0, 0.04, 1, 1))
    kanvas.savefig(GAMBAR, dpi=200)


def cetak_ringkasan(alat, nama, hasil):
    p, k = hasil["puncak"], hasil["korelasi_terkuat"]
    print(f"{alat} {nama:5s} {p['bentuk']:6s} median {p['median_volt']:+.2f} V rentang {p['rentang_volt']}"
          f" pada {p['jumlah_level_berpuncak']}/{p['jumlah_level']} level,"
          f" spearman tinggi-kadar {p['spearman_tinggi_lawan_kadar_icp']:.4f}"
          f" | korelasi arus-kadar terkuat {k['potensial_volt']:+.2f} V ({k['spearman']:.4f})")


def main():
    kadar_icp = kadar_icp_per_label_asli()
    sumbu, cabang = sumbu_potensial()
    shap = json.loads(SHAP.read_text())
    potensial_semua = {nama: sumbu[cabang == kode] for kode, nama in NAMA_CABANG.items()}
    hasil, rata_semua = {}, {}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        hasil[alat], rata_semua[alat] = {}, {}
        for kode, nama in NAMA_CABANG.items():
            indeks = np.flatnonzero(cabang == kode)
            hasil[alat][nama], rata_semua[alat][nama] = analisis_cabang(X, level, kadar, indeks, sumbu[indeks])
            cetak_ringkasan(alat, nama, hasil[alat][nama])
    KELUARAN.write_text(json.dumps(hasil, indent=1))
    gambar(hasil, rata_semua, potensial_semua, shap)


if __name__ == "__main__":
    main()
