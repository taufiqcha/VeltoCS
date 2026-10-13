import json

import matplotlib
import numpy as np

from data import ALAT, HASIL, SCAN, sumbu_potensial
from gambar import (ABU, BIRU, FOLDER, HALAMAN, MERAH, NAMA_ALAT, NAMA_SCAN, ORANYE, ORANYE_ISI, TEKS, desimal,
                    gambar_matriks, plt, simpan)
from uji_puncak import NAMA_CABANG

TEKS_SUPLEMEN = {
    "id": {"scan": "scan ke-", "arus": "arus (satuan alat)", "potensial": "potensial (V terhadap Ag/AgCl)",
           "selisih": "selisih relatif terhadap\nmedian scan 3–10 (%)", "dibuang": "scan 1–2 (dibuang)", "dipakai": "scan 3–10",
           "rho": "korelasi Spearman ρ", "shap": "porsi SHAP (%)", "terkuat": "korelasi terkuat", "contoh": "pengukuran contoh {} mg/L"},
    "en": {"scan": "scan number", "arus": "current (instrument units)", "potensial": "potential (V vs Ag/AgCl)",
           "selisih": "relative difference from\nmedian of scans 3–10 (%)", "dibuang": "scans 1–2 (discarded)", "dipakai": "scans 3–10",
           "rho": "Spearman ρ", "shap": "SHAP share (%)", "terkuat": "strongest correlation", "contoh": "example measurement at {} mg/L"},
}


def gambar_scan(hasil, bahasa):
    T = TEKS_SUPLEMEN[bahasa]
    kanvas, panel = plt.subplots(1, 3, figsize=(HALAMAN, HALAMAN * 0.36))
    for ax, alat in zip(panel, ALAT):
        arsip = np.load(SCAN / f"scans_{alat}.npz")
        n = hasil[alat]["scan"]["contoh"]
        biru = matplotlib.colormaps["Blues"](np.linspace(0.45, 1, 8))
        for s in range(arsip["V"].shape[1]):
            warna, tebal = (ORANYE if s < 2 else biru[s - 2]), (1.1 if s < 2 else 0.7)
            ax.plot(arsip["V"][n, s], arsip["I"][n, s], color=warna, lw=tebal, ls="--" if s == 0 else "-")
        ax.set_title(f"({'ab'[ALAT.index(alat)]}) {NAMA_ALAT[alat]}")
        ax.set_xlabel(T["potensial"])
        ax.set_ylabel(T["arus"])
    ax = panel[2]
    indeks = np.arange(1, 11)
    for alat, warna, penanda in zip(ALAT, (BIRU, ABU), ("o", "s")):
        ax.plot(indeks, hasil[alat]["scan"]["selisih_median_persen"], color=warna, marker=penanda, ms=3.5, lw=1, label=NAMA_ALAT[alat])
    ax.axvspan(0.5, 2.5, color=ORANYE_ISI, lw=0)
    ax.set_xticks(indeks)
    ax.set_xlabel(T["scan"])
    ax.set_ylabel(T["selisih"])
    ax.set_title("(c)")
    ax.legend(frameon=False)
    garis = [plt.Line2D([], [], color=ORANYE, lw=1.1, label=T["dibuang"]), plt.Line2D([], [], color=BIRU, lw=0.9, label=T["dipakai"])]
    kanvas.legend(handles=garis, loc="lower center", ncol=2, frameon=False)
    kanvas.tight_layout(rect=(0, 0.08, 1, 1))
    simpan(kanvas, "S1", bahasa)


def gambar_korelasi(hasil, shap, bahasa):
    T = TEKS_SUPLEMEN[bahasa]
    sumbu, cabang = sumbu_potensial()
    kanvas, panel = plt.subplots(2, 2, figsize=(HALAMAN, HALAMAN * 0.62), sharex=True)
    for baris, alat in enumerate(ALAT):
        rho = np.array(hasil[alat]["spearman_per_potensial"])
        porsi = 100 * np.array(shap[alat]["kepentingan_per_fitur"])
        terkuat = int(np.argmax(np.abs(rho)))
        for kolom, (kode, nama) in enumerate(NAMA_CABANG.items()):
            indeks = np.flatnonzero(cabang == kode)
            ax = panel[baris, kolom]
            kanan = ax.twinx()
            kanan.bar(sumbu[indeks], porsi[indeks], width=0.009, color=ORANYE_ISI, edgecolor=ORANYE, lw=0.3)
            kanan.set_ylim(0, 1.15 * porsi.max())
            kanan.set_ylabel(T["shap"], color=ORANYE)
            kanan.spines["right"].set_visible(True)
            ax.set_zorder(kanan.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.plot(sumbu[indeks], rho[indeks], color=BIRU, lw=1.2)
            ax.axhline(0, color=ABU, lw=0.5)
            if terkuat in indeks:
                ax.axvline(sumbu[terkuat], color=MERAH, ls="--", lw=0.9, label=T["terkuat"])
                ax.legend(frameon=False, loc="upper left")
            ax.set_ylim(-1, 1)
            ax.set_ylabel(T["rho"], color=BIRU)
            ax.set_title(f"({'abcd'[2 * baris + kolom]}) {NAMA_ALAT[alat]}, {NAMA_SCAN[nama]}")
    for ax in panel[1]:
        ax.set_xlabel(T["potensial"])
    kanvas.tight_layout()
    simpan(kanvas, "S3", bahasa)


def main():
    hasil = json.loads((HASIL / "uji_suplemen.json").read_text())
    shap = json.loads((HASIL / "uji_shap.json").read_text())
    for bahasa in FOLDER:
        FOLDER[bahasa].mkdir(parents=True, exist_ok=True)
        gambar_scan(hasil, bahasa)
        gambar_korelasi(hasil, shap, bahasa)
        gambar_matriks({alat: np.array(hasil[alat]["matriks_label_nominal"]) for alat in ALAT}, bahasa, "S2")
        print(bahasa, sorted(p.name for p in FOLDER[bahasa].glob("*_S*.png")))


if __name__ == "__main__":
    main()
