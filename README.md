# VeltoCS

Code for *VeltoCS: Voltammetric electronic tongue for semi-quantitative cadmium screening based on ICP-OES analysis*.

Cyclic voltammograms of Cd²⁺ recorded on a low-cost (LecSens) and a commercial (EmStat4) potentiostat are labelled with ICP-OES concentrations, classified into four decade-wide screening bands, and compared with ICP-OES by Passing–Bablok regression, Bland–Altman analysis, paired tests, recovery, weighted kappa, and false negative and false positive rates at each threshold. `results/` holds the computed results behind every number in the paper. Identifiers in the code are in Indonesian.

## Data

- Raw voltammograms: `01_raw_voltammograms.zip` on Zenodo, [doi:10.5281/zenodo.22236055](https://doi.org/10.5281/zenodo.22236055).
- ICP-OES concentrations (duplicate measurements per level): `data/icp_oes.csv` (Table 1).

## Run

Python 3.11.

```bash
pip install -r requirements.txt
cd kode
python3 pra_proses.py /path/to/01_raw_voltammograms.zip
python3 uji_5fold.py
python3 uji_kesesuaian.py
python3 uji_berpasangan.py
python3 uji_label_nominal.py
python3 uji_puncak.py
python3 uji_suplemen.py
python3 gambar.py
python3 gambar_suplemen.py
```

`uji_tabpfn.py` and `uji_shap.py` run in a separate environment from `requirements-sota.txt`, because TabPFN and SHAP need newer versions of numpy and scikit-learn than the other scripts. TabPFN uses the version 2 weights, which download without an account.

## Scripts

| Script | Result file | Paper |
|---|---|---|
| `uji_5fold.py` | `uji_5fold.json` | Table 3, band accuracy |
| `uji_tabpfn.py` | `uji_tabpfn.json` | Table 3 (TabPFN) |
| `uji_kesesuaian.py` | `uji_kesesuaian.json` | Passing–Bablok, Bland–Altman, kappa, Table 4 |
| `uji_berpasangan.py` | `uji_berpasangan.json` | Paired tests, recovery, difference between potentiostats |
| `uji_label_nominal.py` | `uji_label_nominal.json` | Table 2, label-source ablation |
| `uji_puncak.py` | `uji_puncak.json` | Peak positions and current–concentration correlation |
| `uji_shap.py` | `uji_shap.json` | SHAP windows |
| `uji_suplemen.py` | `uji_suplemen.json` | Scan stability, correlation at every potential, nominal-label confusion matrices |
| `gambar.py` | `figures/Figure_2–4` | Figs. 3–5 |
| `gambar_suplemen.py` | `figures/Figure_S1–S3` | Figs. S1–S3 |

Tables S1 and S2 come from `uji_5fold.json` and `uji_tabpfn.json`. Figures 1 and 2 of the paper are diagrams drawn in draw.io and are not generated here.
