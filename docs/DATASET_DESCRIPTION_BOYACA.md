# Dataset didáctico sintético: calle Boyacá, Guayaquil (2026)

## Propósito

`data/traffic_observations_boyaca_guayaquil_2026.csv` es un recurso sintético de uso didáctico para ejecutar de forma reproducible el modelo híbrido LSTM del repositorio. No reemplaza `traffic_data.csv` y no representa mediciones oficiales ni datos certificados por una autoridad municipal.

## Ubicación y periodo simulado

- **Ubicación simulada:** calle Boyacá, Guayaquil, Ecuador.
- **Punto simulado:** `BOYACA_GYE_A1`.
- **Tipo de vía:** arterial urbana de tres carriles en zona comercial.
- **Campaña observacional proyectada:** del 14 de abril al 1 de mayo de 2026.
- **Serie incluida:** 500 registros consecutivos desde `2026-04-14 00:00:00` hasta `2026-04-19 04:45:00`, dentro de la ventana proyectada.
- **Frecuencia:** un registro cada 15 minutos.

## Criterios de síntesis

La demanda combina variación diaria, perturbaciones deterministas y episodios simulados de lluvia e incidentes. Los máximos de flujo se concentran especialmente entre 11:00 y 12:00 y entre 17:00 y 18:30. El generador mantiene coherencia básica: al aumentar el flujo y la cola, disminuye la velocidad promedio y aumenta el tiempo de recorrido.

## Diccionario de variables

| Variable | Tipo | Unidad / valores | Descripción |
|---|---|---|---|
| `timestamp` | fecha-hora | `YYYY-MM-DD HH:MM:SS` | Inicio del intervalo simulado. |
| `site_id` | categórica | `BOYACA_GYE_A1` | Identificador constante del punto simulado. |
| `flow_total` | entera | vehículos/intervalo | Flujo total estimado durante 15 minutos. |
| `motos` | entera | vehículos/intervalo | Motocicletas incluidas en el flujo total. |
| `buses` | entera | vehículos/intervalo | Buses incluidos en el flujo total. |
| `camiones` | entera | vehículos/intervalo | Camiones incluidos en el flujo total. |
| `velocidad_prom` | decimal | km/h | Velocidad promedio simulada. |
| `tiempo_recorrido` | entera | segundos | Tiempo de recorrido simulado del segmento. |
| `cola` | entera | vehículos | Longitud simulada de la cola. |
| `incidente` | binaria | 0=no, 1=sí | Incidente sintético durante el intervalo. |
| `lluvia` | binaria | 0=no, 1=sí | Lluvia sintética durante el intervalo. |
| `tipo_via` | categórica | `arterial` | Clasificación funcional de la vía. |
| `carriles` | entera | `3` | Número de carriles del segmento simulado. |
| `zona` | categórica | `comercial` | Uso predominante del entorno simulado. |
| `hora_pico` | binaria | 0=no, 1=sí | Marca 07:00–09:00, 11:00–12:00 y 17:00–18:30. |
| `feriado` | binaria | `0` | La simulación no modela feriados. |
| `congestion` | ordinal | 0–4 | Nivel sintético de congestión. |

## Clases de congestión

- **0 — libre:** circulación fluida, flujo bajo y cola mínima.
- **1 — baja:** aumento leve de demanda con operación estable.
- **2 — moderada:** reducción visible de velocidad y crecimiento de cola.
- **3 — alta:** flujo elevado, cola importante y mayor tiempo de recorrido.
- **4 — crítica:** máximos de demanda o perturbaciones severas simuladas.

## Limitaciones

1. Todos los valores son sintéticos y proceden de reglas, no de aforos de campo.
2. La serie representa un solo punto y no describe toda la calle Boyacá ni Guayaquil.
3. Los incidentes y la lluvia son simplificaciones didácticas.
4. Las clases no equivalen a niveles de servicio oficiales.
5. La regularidad del generador puede facilitar patrones que no existirían con datos reales.
6. El dataset no debe emplearse para decisiones operativas, regulación o planificación urbana.

## Uso educativo y abierto

El archivo se publica para docencia, pruebas de software, experimentación reproducible y colaboración de código abierto. Toda presentación o trabajo derivado debe identificarlo explícitamente como **sintético y no oficial**.
