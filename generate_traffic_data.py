"""Genera un dataset sintético de tráfico para entrenar el modelo híbrido LSTM."""

import importlib
import sys


def load_dependencies():
    missing=[]
    modules={}
    for module,package in (("pandas","pandas"),("numpy","numpy")):
        try:
            modules[module]=importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        print("❌ Dependencias faltantes:", ", ".join(missing))
        print("Instala con: pip install -r requirements.txt")
        sys.exit(1)
    return modules


def main() -> None:
    modules=load_dependencies()
    pd=modules["pandas"]
    np=modules["numpy"]

    np.random.seed(42)

    rows = []
    base_time = pd.Timestamp("2026-01-01 06:00:00")

    for i in range(100):
        timestamp = base_time + pd.Timedelta(minutes=15 * i)

        hora = timestamp.hour
        hora_pico = 1 if (7 <= hora <= 9 or 11 <= hora <= 13 or 17 <= hora <= 19) else 0

        base_flow = np.random.randint(40, 80)
        if hora_pico:
            base_flow += np.random.randint(40, 80)

        velocidad = max(10, 45 - (base_flow * 0.15))
        tiempo = int(80 + base_flow * 1.2)
        cola = int(base_flow / 6)

        incidente = np.random.choice([0, 1], p=[0.9, 0.1])
        lluvia = np.random.choice([0, 1], p=[0.85, 0.15])

        if incidente:
            velocidad *= 0.8
            tiempo *= 1.3
            cola += 5

        if base_flow < 80:
            congestion = 0
        elif base_flow < 120:
            congestion = 1
        else:
            congestion = 2

        rows.append(
            [
                timestamp,
                "Boyaca",
                base_flow,
                int(base_flow * 0.3),
                int(base_flow * 0.05),
                int(base_flow * 0.03),
                round(velocidad, 2),
                tiempo,
                cola,
                incidente,
                lluvia,
                "arterial",
                3,
                "comercial",
                hora_pico,
                0,
                congestion,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "site_id",
            "flow_total",
            "motos",
            "buses",
            "camiones",
            "velocidad_prom",
            "tiempo_recorrido",
            "cola",
            "incidente",
            "lluvia",
            "tipo_via",
            "carriles",
            "zona",
            "hora_pico",
            "feriado",
            "congestion",
        ],
    )

    df.to_csv("traffic_data.csv", index=False)
    print("CSV generado correctamente en: traffic_data.csv")


if __name__ == "__main__":
    main()
