# Reproducibility guide

## Educational purpose

This repository is an open educational computational resource for training, evaluating, and interpreting a hybrid Long Short-Term Memory (LSTM) model for synthetic urban congestion prediction. The reproducible workflow is intended to support transparent experimentation, technical instruction, and methodological discussion rather than operational deployment.

## Repository versioning

- **v1.0.0** archived the initial synthetic dataset, hybrid LSTM implementation, and methodological documentation.
- **v1.1.0** adds the reproducible Google Colab notebook, trained-model artifact structure, class-sensitive evaluation outputs, baseline comparison structure, five-class and three-class scenarios, and artifact export workflow.

## How to run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload or open `notebooks/colab_hybrid_lstm_boyaca_guayaquil_reproducible.ipynb`.
3. Select **Runtime > Change runtime type > GPU**.
4. Run all cells in order.
5. Download the generated artifact package.

The notebook clones or opens the repository, audits the runtime, validates the dataset, performs exploratory analysis, applies chronological splitting, trains baseline and hybrid LSTM models, evaluates five-class and three-class scenarios, and exports the generated files. Output files are produced only after the notebook has been executed; no precomputed metrics are stored in this repository.

## Expected outputs

- `outputs/model_comparison_results.csv`
- `outputs/confusion_matrix_hybrid_lstm_5_classes.csv`
- `outputs/confusion_matrix_hybrid_lstm_3_classes.csv`
- `outputs/history_hybrid_lstm_5_classes.csv`
- `outputs/history_hybrid_lstm_3_classes.csv`
- `outputs/environment_summary.json`
- `models/hybrid_lstm_5_classes.keras`
- `models/hybrid_lstm_3_classes.keras`
- `artifacts/boyaca_guayaquil_hybrid_lstm_artifacts.zip`

The generated artifact package also contains fitted preprocessors and scenario metadata needed to reproduce inference.

## Methodological note

The dataset is synthetic, didactic, and non-official. It must not be interpreted as an official municipal traffic measurement or as an operational traffic-management system.
