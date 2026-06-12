"""Ejecuta el modelo híbrido LSTM con el dataset didáctico de proyección de tráfico.

Este archivo no reemplaza el dataset predeterminado. Solo configura el modelo para usar:
    data/proyeccion_trafico_didactica_2026-04-14_a_2026-05-01.csv

Uso:
    python run_didactic_projection_dataset.py
"""

from hybrid_lstm_traffic import CFG, main


CFG.csv_path = "data/proyeccion_trafico_didactica_2026-04-14_a_2026-05-01.csv"


if __name__ == "__main__":
    main()
