# Dataset didáctico de proyección de tráfico

## Nombre del archivo

`data/proyeccion_trafico_didactica_2026-04-14_a_2026-05-01.csv`

## Propósito

Este dataset se incorpora como un recurso didáctico y experimental para probar el modelo híbrido Long Short-Term Memory (LSTM) de predicción de congestión urbana. Su finalidad es permitir la ejecución del flujo de entrenamiento con una serie temporal más amplia que el archivo de ejemplo inicial.

## Cobertura temporal

- Fecha inicial: 2026-04-14 00:00:00
- Fecha final: 2026-05-01 23:45:00
- Intervalo: 15 minutos
- Total de registros: 1728 observaciones

## Contexto de aplicación

El dataset fue estructurado para el trabajo metodológico sobre Smart Cities, aprendizaje automático y movilidad urbana aplicado a Guayaquil, Ecuador. Debe leerse como un insumo de simulación/proyección didáctica para validar el funcionamiento computacional del modelo y su potencial uso en escenarios de baja infraestructura tecnológica.

## Estructura de variables

| Variable | Descripción |
|---|---|
| timestamp | Fecha y hora de observación/proyección |
| site_id | Identificador del punto de observación o segmento urbano |
| flow_total | Flujo vehicular total estimado por intervalo |
| motos | Número estimado de motocicletas |
| buses | Número estimado de buses |
| camiones | Número estimado de camiones |
| velocidad_prom | Velocidad promedio estimada |
| tiempo_recorrido | Tiempo de recorrido estimado |
| cola | Longitud de cola estimada |
| incidente | Presencia de incidente vial, 0 no, 1 sí |
| lluvia | Condición de lluvia, 0 no, 1 sí |
| tipo_via | Tipo de vía |
| carriles | Número de carriles |
| zona | Zona urbana dominante |
| hora_pico | Condición de hora pico, 0 no, 1 sí |
| feriado | Condición de feriado, 0 no, 1 sí |
| congestion | Clase objetivo de congestión |

## Clases de congestión

| Valor | Clase |
|---|---|
| 0 | Congestión baja |
| 1 | Congestión media |
| 2 | Congestión alta |

## Uso recomendado

Para ejecutar el modelo con este dataset:

```bash
python run_didactic_projection_dataset.py
```

También puede utilizarse directamente como entrada si el script principal se extiende con soporte CLI para rutas CSV personalizadas.

## Limitaciones

Este dataset no debe presentarse como registro oficial de monitoreo municipal ni como medición continua institucional. Su uso es didáctico, experimental y de reproducibilidad computacional. Para un conference paper, debe describirse como dataset de proyección/simulación didáctica, salvo que se complemente con evidencia documental de una campaña observacional real.
