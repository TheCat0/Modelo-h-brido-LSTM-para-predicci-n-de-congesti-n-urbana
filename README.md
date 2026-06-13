# Modelo-h-brido-LSTM-para-predicci-n-de-congesti-n-urbana
Se formuló un modelo híbrido de predicción de congestión urbana basado en redes Long Short-Term Memory (LSTM), compuesto por una rama secuencial y una rama contextual. 

## Reproducible educational notebook

The reproducible Google Colab notebook provides an educational workflow to:

- load the synthetic Boyacá–Guayaquil dataset;
- audit the execution environment;
- perform exploratory data analysis;
- diagnose class imbalance;
- construct chronological temporal sequences;
- train the hybrid LSTM model;
- evaluate class-sensitive metrics;
- compare the hybrid model with baseline models;
- run five-class and three-class congestion scenarios; and
- export models, preprocessors, metrics, histories, confusion matrices, metadata, and a traceable artifact package.

No precomputed metrics, confusion matrices, or trained models are included. These files are generated only after executing the notebook.

- [Reproducible Colab notebook](notebooks/colab_hybrid_lstm_boyaca_guayaquil_reproducible.ipynb)
- [Reproducibility guide](docs/reproducibility.md)
- [Release notes for v1.1.0](docs/release_notes_v1.1.0.md)
