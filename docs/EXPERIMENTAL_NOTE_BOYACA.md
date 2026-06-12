# Nota experimental — dataset sintético de Boyacá, Guayaquil

## Pregunta didáctica

¿Puede el modelo híbrido LSTM aprender la relación temporal y contextual entre flujo, velocidad, cola, tiempo de recorrido y cinco niveles ordinales de congestión en una serie sintética reproducible?

## Datos

El experimento utiliza `data/traffic_observations_boyaca_guayaquil_2026.csv`: 500 observaciones sintéticas consecutivas, separadas por 15 minutos, desde el 14 hasta el 19 de abril de 2026. La serie está situada dentro de una campaña observacional proyectada entre el 14 de abril y el 1 de mayo de 2026 para la calle Boyacá, Guayaquil.

El archivo conserva las 17 columnas requeridas por el pipeline y emplea las clases ordinales `0` (libre), `1` (baja), `2` (moderada), `3` (alta) y `4` (crítica).

## Ejecución

```bash
python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
```

## Diseño del pipeline

1. Ordenar por `site_id` y `timestamp`.
2. Construir ventanas temporales de ocho observaciones.
3. Escalar las variables secuenciales con estadísticas del entrenamiento.
4. Procesar variables contextuales numéricas y categóricas.
5. Dividir las ventanas en entrenamiento, validación y prueba con estratificación.
6. Entrenar la rama LSTM y la rama contextual con parada temprana.
7. Reportar pérdida, exactitud, matriz de confusión y métricas por clase.

## Interpretación

La ejecución prueba integración, preprocesamiento y capacidad de aprendizaje sobre datos controlados. Una métrica elevada no demuestra desempeño en tráfico real, porque las relaciones fueron definidas por las reglas de síntesis.

## Amenazas a la validez

- Las ventanas consecutivas comparten observaciones y no son independientes.
- Una división aleatoria puede ubicar patrones muy similares en entrenamiento y prueba.
- Los picos, incidentes y lluvia son simulaciones simplificadas.
- Solo se representa un punto vial y una geometría fija.
- Las cinco clases no han sido calibradas contra una autoridad o sensor real.
- No existe validación externa con mediciones municipales.

Para investigación formal se recomienda recolectar datos reales, separar por jornadas o bloques cronológicos y validar los umbrales con especialistas locales.

## Propósito educativo y abierto

El dataset y los resultados derivados deben presentarse como **sintéticos, didácticos y no oficiales**. No deben utilizarse por sí solos para diagnóstico operativo, planificación vial ni decisiones públicas.
