# Cognitive Distortion Detection & Mental Health NLP Pipeline

An advanced multi-stage machine learning pipeline for cognitive distortion detection and mental health text analysis. This document explicitly breaks down the pipeline, keeping notebooks, datasets, and models separate per stage and individual notebook context.


## Stage I: Initial Fine-Tuning & Baseline Training

### 1. Notebook: `catch_the_st_1ModernBert`
* **Model:** ModernBERT-Large
* **Datasets:**
  * [cbt_longContext_synthetic_datset](https://www.kaggle.com/datasets/arafahmed98229/cbt-longcontext-synthetic-datset/data)
  * [Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset)
  * [Synthetic Cognitive Distortion Dataset](https://www.kaggle.com/datasets/arafahmed98/df-masterno-aug/versions/1/data)
  * [Synthetic Cognitive Distortion Dataset (SCDD-2.2k)](https://www.kaggle.com/datasets/arafahmed98229/synthetic-cognitive-distortion-dataset-scdd-2-2k)
  * [mindReading_and_fortuneTelling_data_synthetic](https://www.kaggle.com/datasets/arafahmed98229/mindreading-and-fortunetelling-data-synthetic)
  * **Optional:**
    * [taf_synthetic](https://www.kaggle.com/datasets/arafahmed98229/taf-synthetic)

### 2. Notebook: `catch_the_st_1MentalRobertalarge`
* **Model:** Mental-RoBERTa-Large
* **Datasets:**
  * [Synthetic Cognitive Distortion Dataset](https://www.kaggle.com/datasets/arafahmed98/df-masterno-aug/versions/1)

---


## Stage II: MLM Domain Adaptation, Stage III: Advanced Stages,
## and Stage IV Ensembling
* **Context:** Masked Language Modeling (MLM) domain adaptation phase utilizing the Reddit Mental Health dataset uploaded directly from local storage (though also available on Kaggle via the original author link).
* **Dataset:**
  * [Reddit Mental Health Dataset (RMHD)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)
### 1. Notebook: `mentalRobertaStagesAndEnsemble`
* **Datasets:**
  * [counselchat_data](https://www.kaggle.com/datasets/weiting016/counselchat-data)
  * [Therapist Q&A](https://www.kaggle.com/datasets/arnmaud/therapist-qa)
  * [Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset)
  * [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/arafahmed99/reddit-mental-health-dataset)
  * [Reddit Mental Health Dataset (original author)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)
  * [ensemble_modernBert_pMulty](https://www.kaggle.com/datasets/arafahmed98/ensemble-modernbert-pmulty)
  * [pMentalRoberta](https://www.kaggle.com/datasets/arafahmed98/pmentalroberta/data)
  * [frozen_ensemble_artifacts](https://www.kaggle.com/datasets/arafahmed98/frozen-ensemble-artifacts/data)
* **Models & Artifacts:**
  * [ensemble_mental_roberta](https://www.kaggle.com/models/arafahmed98/ensemble-mental-roberta/transformers/vv/1)
  * [mental-roberta-stage2-distortion](https://www.kaggle.com/models/arafahmed98/mental-roberta-stage2-distortion/transformers/default/1)
  * [modernBert_pure_multy_phase12_hierarchical_mtl_en](https://www.kaggle.com/models/arafahmed98/pmultymodernbert/Transformers/v2/1)
  * [st1_model_v2_menRoLar](https://www.kaggle.com/models/arafahmed98/st1-model-v2-menrolar/Transformers/default/1)

### 2. Notebook: `modernbertstages.ipynb`
* **Datasets:**
  * [split_from_roberta](https://www.kaggle.com/datasets/arafahmed99/split-from-roberta/data)
  * [Therapist Q&A](https://www.kaggle.com/datasets/arnmaud/therapist-qa)
  * [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/arafahmed99/reddit-mental-health-dataset)
  * [Reddit Mental Health Dataset (original author)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)
  * [counselchat_data](https://www.kaggle.com/datasets/weiting016/counselchat-data)
  * [Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset)
* **Models & Artifacts:**
  * [strong semantics m (modernbert stage 2 strong semantic)](https://www.kaggle.com/models/arafahmed99/strong-semantics-m/Transformers/default/2)
  * [st1_model (modernbert)](https://www.kaggle.com/models/arafahmed98/st1-model/transformers/default/1)
  * [mental-roberta-stage2-distortion](https://www.kaggle.com/models/arafahmed98/mental-roberta-stage2-distortion/Transformers/default/1)




---

## Prescreening

### 1. Notebook: `with roberta`
* **Model:** Roberta Large
* **Datasets:**
  * [For Stage I: Synthetic Cognitive Distortion Dataset (SCDD-2.2k)](https://www.kaggle.com/datasets/arafahmed98229/synthetic-cognitive-distortion-dataset-scdd-2-2k)
  * [For Stage II: Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset)

### 2. Notebook: `with debarta`
* **Model:** DeBERTa v3 Large
* **Datasets:**
  * [Synthetic Cognitive Distortion Dataset (SCDD-2.2k)](https://www.kaggle.com/datasets/arafahmed98229/synthetic-cognitive-distortion-dataset-scdd-2-2k)

### 3. Notebook: `with bert`
* **Model:** BERT Large Uncased
* **Datasets:**
  * [Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset)
  * [For Stage I: Synthetic Cognitive Distortion Dataset (SCDD-2.2k)](https://www.kaggle.com/datasets/arafahmed98229/synthetic-cognitive-distortion-dataset-scdd-2-2k)
  * [For Stage II: Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset)

---

