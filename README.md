# Modelo-h-brido-LSTM-para-predicci-n-de-congesti-n-urbana

Se formuló un modelo híbrido de predicción de congestión urbana basado en redes Long Short-Term Memory (LSTM), compuesto por una rama secuencial y una rama contextual.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hybrid_lstm_traffic.py
```

## Solución de alertas comunes

1. **`ModuleNotFoundError` (ej. `numpy`)**
   - Instala dependencias con `pip install -r requirements.txt`.
2. **Error de instalación por proxy (`403 Forbidden`)**
   - Configura `HTTP_PROXY`/`HTTPS_PROXY` o usa un mirror interno antes de correr `pip install`.

El script ahora valida dependencias faltantes y muestra un mensaje claro en lugar de un traceback críptico.
