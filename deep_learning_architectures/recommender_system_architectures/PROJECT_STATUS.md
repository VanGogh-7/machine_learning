# Project Status

| Project | Model | Dataset | Has README | Has config.py | Has data.py | Has model.py | Has train.py | Has predict/recommend/retrieve script | Has prepare_data.py | Debug command |
|---|---|---|---|---|---|---|---|---|---|---|
| `01_wide_deep_ctr_prediction` | Wide & Deep | Criteo-style CTR | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ⚠️ Criteo CSVs expected | `python scripts/train.py` |
| `02_deepfm_ctr_prediction` | DeepFM | Criteo-style CTR | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ⚠️ Criteo CSVs expected | `python scripts/train.py` |
| `03_dcn_ctr_prediction` | DCN | Criteo-style CTR | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ⚠️ Criteo CSVs expected | `python scripts/train.py` |
| `04_xdeepfm_ctr_prediction` | xDeepFM | Criteo-style CTR | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ⚠️ Criteo CSVs expected | `python scripts/train.py` |
| `05_din_ctr_prediction` | DIN | MovieLens 10M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ✅ | `python scripts/prepare_data.py && python scripts/train.py` |
| `06_dien_ctr_prediction` | DIEN | MovieLens 10M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ✅ | `python scripts/prepare_data.py && python scripts/train.py` |
| `07_ncf_collaborative_filtering` | NCF / NeuMF | MovieLens 10M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `predict.py` | ✅ | `python scripts/prepare_data.py && python scripts/train.py` |
| `08_two_tower_retrieval` | Two-Tower Retrieval | MovieLens 10M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `retrieve.py` | ✅ | `python scripts/prepare_data.py && python scripts/train.py` |
| `09_sasrec_sequential_recommendation` | SASRec | MovieLens 10M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `recommend.py` | ✅ | `python scripts/prepare_data.py && python scripts/train.py` |
| `10_lightgcn_recommendation` | LightGCN | MovieLens 10M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `recommend.py` | ✅ | `python scripts/prepare_data.py && python scripts/train.py` |
