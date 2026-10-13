import json

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data import AKAR, ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri, sumbu_potensial
from model import BENIH, KLASIFIER, PITA, REGRESOR, nomor_pita
from uji_5fold import prediksi_log_kadar, prediksi_pita
from uji_kesesuaian import bland_altman_persen, passing_bablok
from uji_puncak import NAMA_CABANG, rata_per_level

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = {"en": AKAR / "figures"}
AWALAN = {"id": "Gambar", "en": "Figure"}
BIRU, BIRU_ISI, ORANYE, ORANYE_ISI = "#1F6FB5", "#EAF3FA", "#E69F00", "#FDF0DC"
ABU, ABU_ISI, MERAH = "#6E6E6E", "#F0F0F0", "#B03A2E"
HALAMAN = 494.51 / 72
NAMA_ALAT = {"LC": "LecSens", "EM": "EmStat4"}
NAMA_PITA = ("< 10", "10–100", "100–1000", "≥ 1000")
NAMA_SCAN = {"maju": "forward scan", "balik": "reverse scan"}
TEKS = {
    "id": {"band_prediksi": "band terprediksi (mg/L)", "band_icp": "band ICP-OES (mg/L)", "identitas": "identity line",
           "loa": "limits of agreement 95%", "kadar_icp": "kadar ICP-OES (mg/L)", "kadar_prediksi": "kadar terprediksi (mg/L)",
           "rerata": "rerata kedua metode (mg/L)", "selisih": "selisih relatif (%)", "arus": "arus (satuan alat)",
           "potensial": "potensial (V terhadap Ag/AgCl)", "puncak": "wilayah puncak", "median": "median letak puncak",
           "shap": "SHAP window terpenting"},
    "en": {"band_prediksi": "predicted band (mg/L)", "band_icp": "ICP-OES band (mg/L)", "identitas": "identity line",
           "loa": "95% limits of agreement", "kadar_icp": "ICP-OES concentration (mg/L)", "kadar_prediksi": "predicted concentration (mg/L)",
           "rerata": "mean of both methods (mg/L)", "selisih": "relative difference (%)", "arus": "current (instrument units)",
           "potensial": "potential (V vs Ag/AgCl)", "puncak": "peak region", "median": "median peak position",
           "shap": "most important SHAP window"},
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def simpan(kanvas, nomor, bahasa):
    for jenis in ("png", "pdf"):
        kanvas.savefig(FOLDER[bahasa] / f"{AWALAN[bahasa]}_{nomor}.{jenis}")
    plt.close(kanvas)


def desimal(teks, bahasa):
    return teks.replace(".", ",") if bahasa == "id" else teks


def matriks_svm(data):
    hasil = {}
    for alat in ALAT:
        X, level, kadar = data[alat]
        pita = nomor_pita(kadar)
        matriks = np.zeros((len(PITA), len(PITA)))
        for benih in BENIH:
            np.add.at(matriks, (pita, prediksi_pita(X, level, pita, KLASIFIER["SVM"], benih)), 1)
        hasil[alat] = matriks
    return hasil


def gambar_matriks(semua_matriks, bahasa):
    T = TEKS[bahasa]
    kanvas, panel = plt.subplots(1, 2, figsize=(HALAMAN, HALAMAN * 0.42))
    peta = matplotlib.colors.LinearSegmentedColormap.from_list("biru", ["#FFFFFF", BIRU])
    for ax, alat in zip(panel, ALAT):
        matriks = semua_matriks[alat]
        persen_baris = 100 * matriks / matriks.sum(axis=1, keepdims=True)
        ax.imshow(persen_baris, cmap=peta, vmin=0, vmax=100)
        for i in range(len(PITA)):
            for j in range(len(PITA)):
                if matriks[i, j]:
                    warna = "white" if persen_baris[i, j] > 60 else "black"
                    ax.text(j, i, desimal(f"{persen_baris[i, j]:.1f}%\n({int(matriks[i, j])})", bahasa),
                            ha="center", va="center", fontsize=7, color=warna)
        ax.set_xticks(range(len(PITA)), NAMA_PITA)
        ax.set_yticks(range(len(PITA)), NAMA_PITA)
        ax.set_xlabel(T["band_prediksi"])
        ax.set_ylabel(T["band_icp"])
        ax.set_title(f"({'ab'[ALAT.index(alat)]}) {NAMA_ALAT[alat]}")
        ax.spines[:].set_visible(True)
    kanvas.tight_layout()
    simpan(kanvas, 3, bahasa)


def prediksi_et(data):
    return {alat: 10 ** prediksi_log_kadar(data[alat][0], data[alat][1], data[alat][2], REGRESOR["ET"], BENIH[0]) for alat in ALAT}


def gambar_kesesuaian(data, semua_prediksi, bahasa):
    T = TEKS[bahasa]
    kanvas, panel = plt.subplots(2, 2, figsize=(HALAMAN, HALAMAN * 0.75))
    for baris, alat in enumerate(ALAT):
        X, level, kadar = data[alat]
        prediksi = semua_prediksi[alat]
        pb, ba = passing_bablok(kadar, prediksi), bland_altman_persen(kadar, prediksi)
        ax = panel[baris, 0]
        ax.scatter(kadar, prediksi, s=9, color=BIRU, alpha=0.6, linewidths=0, zorder=2)
        garis = np.array([kadar.min() * 0.8, kadar.max() * 1.2])
        ax.plot(garis, pb["intersep"] + pb["kemiringan"] * garis, color=MERAH, lw=1.6, label="Passing–Bablok", zorder=3)
        ax.plot(garis, garis, color="black", ls="--", lw=0.9, label=T["identitas"], zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(T["kadar_icp"])
        ax.set_ylabel(T["kadar_prediksi"])
        ax.set_title(f"({'ac'[baris]}) {NAMA_ALAT[alat]}, Passing–Bablok")
        ax.legend(frameon=False, loc="upper left")
        ax = panel[baris, 1]
        rerata = (prediksi + kadar) / 2
        selisih = 100 * (prediksi - kadar) / rerata
        ax.scatter(rerata, selisih, s=9, color=BIRU, alpha=0.6, linewidths=0, zorder=2)
        ax.axhline(ba["bias_persen"], color=MERAH, lw=1.6, label="bias", zorder=3)
        for batas in ba["batas_kesesuaian_persen"]:
            ax.axhline(batas, color=MERAH, lw=1.2, ls="--", zorder=3)
        ax.plot([], [], color=MERAH, lw=1.2, ls="--", label=T["loa"])
        ax.set_xscale("log")
        ax.set_xlabel(T["rerata"])
        ax.set_ylabel(T["selisih"])
        ax.set_title(f"({'bd'[baris]}) {NAMA_ALAT[alat]}, Bland–Altman")
        ax.legend(frameon=False, loc="lower right")
    kanvas.tight_layout()
    simpan(kanvas, 2, bahasa)


def gambar_voltamogram(data, puncak, shap, bahasa):
    T = TEKS[bahasa]
    sumbu, cabang = sumbu_potensial()
    kanvas, panel = plt.subplots(2, 2, figsize=(HALAMAN, HALAMAN * 0.72), sharex=True)
    for baris, alat in enumerate(ALAT):
        X, level, _ = data[alat]
        for kolom, (kode, nama) in enumerate(NAMA_CABANG.items()):
            indeks = np.flatnonzero(cabang == kode)
            rata = rata_per_level(X, level, indeks)
            ax = panel[baris, kolom]
            warna = matplotlib.colormaps["Blues"](np.linspace(0.3, 1, len(rata)))
            for w, (nilai, kurva) in zip(warna, sorted(rata.items())):
                ax.plot(sumbu[indeks], kurva, color=w, lw=0.8)
            for j in [j for j in shap[alat]["jendela_terpenting"] if j["cabang"] == nama][:2]:
                ax.axvspan(j["dari_volt"], j["sampai_volt"], color=ORANYE, alpha=0.35, lw=0)
            pk = puncak[alat][nama]["puncak"]
            ax.axvspan(*pk["rentang_volt"], color=ABU, alpha=0.25, lw=0)
            ax.axvline(pk["median_volt"], color=ABU, ls="--", lw=0.8)
            ax.set_title(f"({'abcd'[2 * baris + kolom]}) {NAMA_ALAT[alat]}, {NAMA_SCAN[nama]}")
            ax.set_ylabel(T["arus"])
    for ax in panel[1]:
        ax.set_xlabel(T["potensial"])
    penanda = [Patch(color=ABU, alpha=0.25, label=T["puncak"]), Line2D([], [], color=ABU, ls="--", lw=0.8, label=T["median"]),
               Patch(color=ORANYE, alpha=0.35, label=T["shap"])]
    kanvas.legend(handles=penanda, loc="lower center", ncol=3, frameon=False)
    kanvas.tight_layout(rect=(0, 0.05, 1, 1))
    simpan(kanvas, 4, bahasa)


def main():
    for folder in FOLDER.values():
        folder.mkdir(parents=True, exist_ok=True)
    kadar_icp = kadar_icp_per_label_asli()
    data = {alat: muat_voltametri(alat, kadar_icp) for alat in ALAT}
    semua_matriks, semua_prediksi = matriks_svm(data), prediksi_et(data)
    puncak = json.loads((HASIL / "uji_puncak.json").read_text())
    shap = json.loads((HASIL / "uji_shap.json").read_text())
    for bahasa in FOLDER:
        gambar_kesesuaian(data, semua_prediksi, bahasa)
        gambar_matriks(semua_matriks, bahasa)
        gambar_voltamogram(data, puncak, shap, bahasa)
        print(bahasa, sorted(p.name for p in FOLDER[bahasa].glob("*.p*")))


if __name__ == "__main__":
    main()
