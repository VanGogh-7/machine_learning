# Run All Projects in Debug Mode

These commands assume you are starting from the repository root:

```text
machine_learning/
```

Do not disable `debug_mode` until the debug run works.

## 01 Wide & Deep

The processed Criteo CSV files must already exist under `datasets/criteo_ctr/`.

```bash
cd deep_learning_architectures/recommender_system_architectures/01_wide_deep_ctr_prediction
python scripts/train.py
python scripts/predict.py
```

## 02 DeepFM

The processed Criteo CSV files must already exist under `datasets/criteo_ctr/`.

```bash
cd deep_learning_architectures/recommender_system_architectures/02_deepfm_ctr_prediction
python scripts/train.py
python scripts/predict.py
```

## 03 DCN

The processed Criteo CSV files must already exist under `datasets/criteo_ctr/`.

```bash
cd deep_learning_architectures/recommender_system_architectures/03_dcn_ctr_prediction
python scripts/train.py
python scripts/predict.py
```

## 04 xDeepFM

The processed Criteo CSV files must already exist under `datasets/criteo_ctr/`.

```bash
cd deep_learning_architectures/recommender_system_architectures/04_xdeepfm_ctr_prediction
python scripts/train.py
python scripts/predict.py
```

## 05 DIN

```bash
cd deep_learning_architectures/recommender_system_architectures/05_din_ctr_prediction
python scripts/prepare_data.py
python scripts/train.py
python scripts/predict.py
```

## 06 DIEN

```bash
cd deep_learning_architectures/recommender_system_architectures/06_dien_ctr_prediction
python scripts/prepare_data.py
python scripts/train.py
python scripts/predict.py
```

## 07 NCF

```bash
cd deep_learning_architectures/recommender_system_architectures/07_ncf_collaborative_filtering
python scripts/prepare_data.py
python scripts/train.py
python scripts/predict.py
```

## 08 Two-Tower

```bash
cd deep_learning_architectures/recommender_system_architectures/08_two_tower_retrieval
python scripts/prepare_data.py
python scripts/train.py
python scripts/build_item_index.py
python scripts/retrieve.py
```

## 09 SASRec

```bash
cd deep_learning_architectures/recommender_system_architectures/09_sasrec_sequential_recommendation
python scripts/prepare_data.py
python scripts/train.py
python scripts/recommend.py
```

## 10 LightGCN

```bash
cd deep_learning_architectures/recommender_system_architectures/10_lightgcn_recommendation
python scripts/prepare_data.py
python scripts/train.py
python scripts/recommend.py
```
