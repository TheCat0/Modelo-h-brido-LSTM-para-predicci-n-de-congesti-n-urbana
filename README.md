# Modelo híbrido LSTM para predicción de congestión urbana

Este proyecto implementa un modelo híbrido para la predicción de congestión urbana basado en redes de memoria a corto y largo plazo (Long Short-Term Memory, LSTM). La arquitectura combina una rama secuencial, que procesa la evolución temporal del tráfico, y una rama contextual, que incorpora características de la vía y del entorno.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_traffic_data.py
python hybrid_lstm_traffic.py
```

También se puede seleccionar el conjunto de datos sintético de la calle Boyacá, Guayaquil, sin alterar el comportamiento predeterminado:

```bash
python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
```

## Solución de alertas comunes

1. **`ModuleNotFoundError` (ej. `numpy`)**
   - Instala dependencias con `pip install -r requirements.txt`.
2. **Error de instalación causado por un proxy (`403 Forbidden`)**
   - Configura `HTTP_PROXY`/`HTTPS_PROXY` o utiliza un repositorio espejo interno antes de ejecutar `pip install`.

El script ahora valida dependencias faltantes y muestra un mensaje claro en lugar de un traceback críptico.

### Ejecución cuando la red local bloquea PyPI

El repositorio incluye el flujo de trabajo manual **Entrenar modelo LSTM de tráfico** en GitHub Actions. Este permite instalar las dependencias y entrenar en una red externa sin modificar los conjuntos de datos del proyecto.

1. Abre la pestaña **Actions** del repositorio en GitHub.
2. Selecciona **Entrenar modelo LSTM de tráfico**.
3. Pulsa el botón **Run workflow**.
4. Selecciona `both` para entrenar con `traffic_data.csv` y con el conjunto de datos de Boyacá, o elige solo uno.
5. Espera a que los trabajos terminen y descarga los artefactos `hybrid-lstm-traffic-default` y/o `hybrid-lstm-traffic-boyaca`.

Cada artefacto conserva durante 14 días el modelo, los preprocesadores ajustados, los metadatos y el registro completo del entrenamiento. Si la ejecución falla, comparte el registro de la ejecución para continuar el diagnóstico.

## Evaluación temporal y artefactos reproducibles

La evaluación divide cada `site_id` cronológicamente **antes** de crear las ventanas. Entre entrenamiento, validación y prueba se descartan `time_steps - 1` filas, lo que evita que ventanas de particiones diferentes compartan pasos temporales. No se emplean división aleatoria ni estratificación en las series temporales.

El escalador secuencial y el preprocesador contextual se ajustan exclusivamente con la partición de entrenamiento. Al finalizar, se crea el directorio `artifacts/` con los siguientes archivos:

```text
artifacts/hybrid_lstm_traffic_model.keras
artifacts/sequence_scaler.joblib
artifacts/context_preprocessor.joblib
artifacts/model_metadata.json
```

`model_metadata.json` conserva las columnas, categorías, configuración, longitud de secuencia, rangos cronológicos, conteos de ventanas y métricas de prueba necesarios para reconstruir una inferencia consistente. El conjunto de datos de Boyacá continúa siendo sintético, didáctico y no oficial.

## Conjunto de datos de ejemplo

Se incluye el archivo `traffic_data.csv` en la raíz del proyecto para realizar pruebas rápidas del flujo de procesamiento.

## Conjunto de datos didáctico: observaciones sintéticas en la calle Boyacá, Guayaquil

El archivo `data/traffic_observations_boyaca_guayaquil_2026.csv` es un conjunto de datos **sintético de uso didáctico** que simula 500 observaciones consecutivas, separadas por intervalos de 15 minutos, en la Calle Boyacá de Guayaquil, Ecuador. La serie comienza el 14 de abril de 2026 y representa una campaña observacional proyectada dentro del periodo comprendido entre el 14 de abril y el 1 de mayo de 2026.

Su propósito es permitir pruebas reproducibles del modelo híbrido LSTM y demostrar el flujo de carga, preprocesamiento, entrenamiento y evaluación. Los datos no corresponden a una medición oficial y no deben presentarse como información certificada por una autoridad municipal.

```bash
python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
```

Consulta la [descripción del conjunto de datos](docs/DATASET_DESCRIPTION_BOYACA.md), el [protocolo observacional proyectado](docs/FIELD_OBSERVATION_PROTOCOL_BOYACA.md) y la [nota experimental](docs/EXPERIMENTAL_NOTE_BOYACA.md) antes de interpretar resultados.
