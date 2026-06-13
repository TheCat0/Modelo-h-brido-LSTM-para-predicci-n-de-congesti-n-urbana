# v1.1.0 – Reproducible Colab notebook and trained model artifacts

This release extends v1.0.0 with:

- a reproducible Google Colab notebook fixed to the `v1.1.0` release tag;
- explicit execution-environment and dependency validation;
- explicit synthetic-dataset availability and schema validation;
- hybrid LSTM training with chronological temporal partitions;
- comparison with baseline classification models;
- five-class and three-class congestion scenarios; and
- structured export of metrics, histories, confusion matrices, trained models, fitted preprocessors, metadata, and a downloadable artifact package.

No metrics, confusion matrices, or trained models are included in the repository without a real notebook execution. The dataset remains synthetic, didactic, and non-official. The results should be interpreted as evidence of a reproducible educational experiment, not as operational traffic-management deployment.
