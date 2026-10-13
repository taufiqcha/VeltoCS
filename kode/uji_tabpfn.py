import json

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from data import ALAT, HASIL, kadar_icp_per_label_asli, muat_voltametri
from model import BENIH, PITA, nomor_pita
from uji_5fold import (metrik_kuantifikasi, metrik_semi_kuantifikasi, prediksi_log_kadar,
                       prediksi_pita, rerata_benih)

KELUARAN = HASIL / "uji_tabpfn.json"
VERSI = ModelVersion.V2
PERANGKAT = "mps"


def regresor_tabpfn(benih):
    return TabPFNRegressor.create_default_for_version(VERSI, device=PERANGKAT, random_state=benih)


def klasifier_tabpfn(benih):
    return TabPFNClassifier.create_default_for_version(VERSI, device=PERANGKAT, random_state=benih)


def main():
    kadar_icp = kadar_icp_per_label_asli()
    keluaran = {"versi": VERSI.value, "perangkat": PERANGKAT, "alat": {}}
    for alat in ALAT:
        X, level, kadar = muat_voltametri(alat, kadar_icp)
        pita = nomor_pita(kadar)
        kuantifikasi = rerata_benih([metrik_kuantifikasi(kadar, prediksi_log_kadar(X, level, kadar, regresor_tabpfn, benih))
                                     for benih in BENIH])
        per_benih = [metrik_semi_kuantifikasi(pita, prediksi_pita(X, level, pita, klasifier_tabpfn, benih))
                     for benih in BENIH]
        semi = {**rerata_benih(per_benih), "matriks_benih_0": per_benih[0]["matriks"]}
        keluaran["alat"][alat] = {"kuantifikasi_icp": kuantifikasi, "semi_kuantifikasi": semi}
        print(f"{alat} TabPFN R2 {kuantifikasi['r2'][0]:.4f}  R2 log {kuantifikasi['r2_log'][0]:.4f}"
              f"  akurasi pita {semi['akurasi'][0]:.4f}  F1 {semi['f1_makro'][0]:.4f}", flush=True)
    keluaran["pita"] = list(PITA)
    KELUARAN.write_text(json.dumps(keluaran, indent=1))


if __name__ == "__main__":
    main()
