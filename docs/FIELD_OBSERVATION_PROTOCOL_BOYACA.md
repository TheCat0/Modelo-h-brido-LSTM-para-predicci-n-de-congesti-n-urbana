# Protocolo proyectado de observación — calle Boyacá

## Alcance

Este documento describe cómo una futura campaña de campo podría recopilar variables compatibles con el dataset sintético `traffic_observations_boyaca_guayaquil_2026.csv`. El CSV incluido en el repositorio es didáctico: no demuestra que esta campaña haya sido realizada ni constituye evidencia municipal oficial.

## Diseño proyectado

- **Corredor:** calle Boyacá, Guayaquil, Ecuador.
- **Punto de referencia:** `BOYACA_GYE_A1`.
- **Ventana propuesta:** 14 de abril a 1 de mayo de 2026.
- **Intervalo de registro:** 15 minutos.
- **Variables:** conteos agregados, velocidad, tiempo de recorrido, cola, incidentes, lluvia y contexto vial.
- **Privacidad:** no registrar matrículas, rostros ni información personal.

## Registro por intervalo

1. Anotar la fecha y hora local al inicio del intervalo.
2. Contar el flujo total y los subconjuntos de motos, buses y camiones.
3. Estimar la velocidad promedio mediante una distancia de referencia estable.
4. registrar el tiempo de recorrido en segundos.
5. Estimar la cola en número de vehículos.
6. Marcar incidentes y lluvia con valores binarios.
7. Registrar tipo de vía, carriles, zona, hora pico y feriado.
8. Asignar la clase de congestión con criterios previamente calibrados.
9. Documentar en una bitácora cualquier interrupción o pérdida de visibilidad.

## Franjas de interés

La campaña debería cubrir todo el día para evitar sesgo de selección y prestar especial atención a:

- 11:00–12:00.
- 17:00–18:30.

El dataset sintético concentra sus máximos de demanda en esas franjas para facilitar pruebas del modelo.

## Escala ordinal propuesta

- `0`: libre.
- `1`: baja.
- `2`: moderada.
- `3`: alta.
- `4`: crítica.

Antes de usar esta escala con observaciones reales, sus umbrales deben calibrarse con especialistas locales y contrastarse con sensores o fuentes independientes.

## Control de calidad

- Sincronizar los relojes de los observadores.
- Realizar doble conteo en una muestra de intervalos.
- Verificar que las categorías vehiculares sean subconjuntos coherentes del flujo total.
- Mantener orden cronológico y frecuencia constante.
- Conservar datos originales y documentar cualquier limpieza.
- Separar claramente datos reales, datos imputados y datos sintéticos.

## Finalidad

Este protocolo y el dataset asociado tienen propósito educativo y de código abierto. No constituyen certificación oficial, estudio de impacto vial ni recomendación de política pública.
