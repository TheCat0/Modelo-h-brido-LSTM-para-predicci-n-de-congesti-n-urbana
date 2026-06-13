# Modelo híbrido LSTM para predicción de congestión urbana

Se formuló un modelo híbrido de predicción de congestión urbana basado en redes Long Short-Term Memory (LSTM), compuesto por una rama secuencial y una rama contextual.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_traffic_data.py
python hybrid_lstm_traffic.py
```

También se puede seleccionar el dataset piloto de Boyacá, Guayaquil, sin cambiar el comportamiento predeterminado:

```bash
python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
```

## Solución de alertas comunes

1. **`ModuleNotFoundError` (ej. `numpy`)**
   - Instala dependencias con `pip install -r requirements.txt`.
2. **Error de instalación por proxy (`403 Forbidden`)**
   - Configura `HTTP_PROXY`/`HTTPS_PROXY` o usa un mirror interno antes de correr `pip install`.

El script ahora valida dependencias faltantes y muestra un mensaje claro en lugar de un traceback críptico.

### Ejecución cuando la red local bloquea PyPI

El repositorio incluye el workflow manual **Entrenar modelo LSTM de tráfico** en GitHub Actions. Este permite instalar las dependencias y entrenar en una red externa sin cambiar los datasets del proyecto.

1. Abre la pestaña **Actions** del repositorio en GitHub.
2. Selecciona **Entrenar modelo LSTM de tráfico**.
3. Pulsa **Run workflow**.
4. Selecciona `both` para entrenar con `traffic_data.csv` y con el dataset de Boyacá, o elige solo uno.
5. Espera a que los trabajos terminen y descarga los artefactos `hybrid-lstm-traffic-default` y/o `hybrid-lstm-traffic-boyaca`.

Cada artefacto conserva durante 14 días el modelo `.keras` y el registro completo del entrenamiento. Si la ejecución falla, comparte el registro del job para continuar el diagnóstico.

## Dataset de ejemplo

Se incluye un `traffic_data.csv` de ejemplo en la raíz del proyecto para pruebas rápidas del flujo.

## Dataset didáctico: observaciones sintéticas en Calle Boyacá, Guayaquil

El archivo `data/traffic_observations_boyaca_guayaquil_2026.csv` es un dataset **sintético de uso didáctico** que simula 500 observaciones consecutivas, separadas por intervalos de 15 minutos, en la Calle Boyacá de Guayaquil, Ecuador. La serie comienza el 14 de abril de 2026 y representa una campaña observacional proyectada dentro del periodo comprendido entre el 14 de abril y el 1 de mayo de 2026.

Su propósito es permitir pruebas reproducibles del modelo híbrido LSTM y demostrar el flujo de carga, preprocesamiento, entrenamiento y evaluación. Los datos no corresponden a una medición oficial y no deben presentarse como información certificada por una autoridad municipal.

```bash
python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
```

Consulta la [descripción del dataset](docs/DATASET_DESCRIPTION_BOYACA.md), el [protocolo observacional proyectado](docs/FIELD_OBSERVATION_PROTOCOL_BOYACA.md) y la [nota experimental](docs/EXPERIMENTAL_NOTE_BOYACA.md) antes de interpretar resultados.

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

No precomputed metrics, confusion matrices, or trained models are included. These files are generated only after executing the notebook successfully in Google Colab.

- [Reproducible Colab notebook](notebooks/colab_hybrid_lstm_boyaca_guayaquil_reproducible.ipynb)
- [Reproducibility guide](docs/reproducibility.md)
- [Release notes for v1.1.0](docs/release_notes_v1.1.0.md)

## Authors

- Jefferson Cabrera-Amaiquema<br>
  Instituto Superior Tecnológico Bolivariano de Tecnología / Universidad Privada Domingo Savio<br>
  ORCID: https://orcid.org/0000-0003-4623-4462

- Renata Cervantes-Avilés<br>
  Instituto Superior Tecnológico Bolivariano de Tecnología / Universidad Bolivariana del Ecuador<br>
  ORCID: https://orcid.org/0009-0001-9898-7477

- Darwin Manzano-Cuenca<br>
  Instituto Superior Tecnológico Bolivariano de Tecnología / Universidad de La Habana<br>
  ORCID: https://orcid.org/0000-0001-9770-1441

- Evelyn López-Segura<br>
  Instituto Superior Tecnológico Bolivariano de Tecnología / Universidad de Guayaquil<br>
  ORCID: https://orcid.org/0009-0003-5451-7937
