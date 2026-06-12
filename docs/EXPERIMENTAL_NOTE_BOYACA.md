# Nota experimental — conjunto de datos sintético de Boyacá, Guayaquil

## Pregunta didáctica

¿Puede el modelo híbrido LSTM aprender la relación temporal y contextual entre flujo, velocidad, cola, tiempo de recorrido y cinco niveles ordinales de congestión en una serie sintética reproducible?

## Datos

El experimento utiliza `data/traffic_observations_boyaca_guayaquil_2026.csv`: 500 observaciones sintéticas consecutivas, separadas por 15 minutos, desde el 14 hasta el 19 de abril de 2026. La serie está situada dentro de una campaña observacional proyectada entre el 14 de abril y el 1 de mayo de 2026 para la calle Boyacá, Guayaquil.

El archivo conserva las 17 columnas requeridas por el flujo de procesamiento y emplea las clases ordinales `0` (libre), `1` (baja), `2` (moderada), `3` (alta) y `4` (crítica).

## Ejecución

```bash
python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
```

## Diseño del flujo de procesamiento

1. Ordenar cada `site_id` por `timestamp`.
2. Dividir las filas cronológicamente en entrenamiento, validación y prueba antes de crear ventanas.
3. Purgar `time_steps - 1` filas entre particiones para que no compartan pasos temporales.
4. Ajustar el escalador secuencial y el preprocesador contextual únicamente con entrenamiento.
5. Transformar cada partición con esos mismos objetos y construir ventanas independientes.
6. Entrenar la rama LSTM y la rama contextual con parada temprana.
7. Evaluar una sola vez sobre el bloque temporal de prueba.
8. Guardar modelo, preprocesadores y metadatos para inferencia reproducible.

## Interpretación

La ejecución prueba integración, preprocesamiento y capacidad de aprendizaje sobre datos controlados. Una métrica elevada no demuestra desempeño en tráfico real, porque las relaciones fueron definidas por las reglas de síntesis.

## Amenazas a la validez

- Las ventanas dentro de una misma partición comparten observaciones y conservan dependencia temporal.
- La partición cronológica purgada evita solapamiento entre entrenamiento, validación y prueba, pero una sola serie sigue ofreciendo evidencia limitada.
- Los picos, incidentes y lluvia son simulaciones simplificadas.
- Solo se representa un punto vial y una geometría fija.
- Las cinco clases no han sido calibradas contra una autoridad o sensor real.
- No existe validación externa con mediciones municipales.

Para investigación formal se recomienda recolectar datos reales, separar por jornadas o bloques cronológicos y validar los umbrales con especialistas locales.

## Propósito educativo y abierto

El conjunto de datos y los resultados derivados deben presentarse como **sintéticos, didácticos y no oficiales**. No deben utilizarse por sí solos para diagnóstico operativo, planificación vial ni decisiones públicas.
