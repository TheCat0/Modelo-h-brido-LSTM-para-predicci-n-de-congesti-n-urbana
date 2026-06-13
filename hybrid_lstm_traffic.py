"""Modelo híbrido LSTM para predicción de congestión urbana.

Requisitos:
    pip install -r requirements.txt

Uso:
    python hybrid_lstm_traffic.py
    python hybrid_lstm_traffic.py --csv data/traffic_observations_boyaca_guayaquil_2026.csv
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple


# =========================
# Configuración
# =========================
@dataclass
class Config:
    csv_path: str = "traffic_data.csv"
    time_steps: int = 8
    batch_size: int = 32
    epochs: int = 50
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2  # respecto al conjunto de entrenamiento
    target_col: str = "congestion"
    timestamp_col: str = "timestamp"
    site_col: str = "site_id"


CFG = Config()


def ensure_dependencies() -> dict[str, Any]:
    """Valida e importa dependencias de forma segura para evitar trazas crípticas."""
    required_modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "tensorflow": "tensorflow",
    }

    loaded: dict[str, Any] = {}
    missing: list[str] = []

    for module_name, package_name in required_modules.items():
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)

    if missing:
        print("❌ Dependencias faltantes:", ", ".join(sorted(set(missing))))
        print("Instala con: pip install -r requirements.txt")
        print(
            "Si tu entorno usa proxy, exporta HTTP_PROXY/HTTPS_PROXY o usa un mirror interno antes de instalar."
        )
        raise SystemExit(1)

    return loaded


# =========================
# Carga de datos
# =========================
def load_data(pd: Any, csv_path: str):
    """Carga el CSV y valida columnas mínimas."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
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
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Hay valores inválidos en la columna 'timestamp'.")

    # Orden temporal por sitio
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    return df


# =========================
# Preprocesamiento
# =========================
def build_preprocessor(
    StandardScaler: Any,
    OneHotEncoder: Any,
    Pipeline: Any,
    ColumnTransformer: Any,
    numeric_context_cols: List[str],
    categorical_context_cols: List[str],
):
    """Preprocesador para variables contextuales."""
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_context_cols),
            ("cat", categorical_pipeline, categorical_context_cols),
        ]
    )
    return preprocessor


def scale_sequence_features(StandardScaler: Any, train_seq, val_seq, test_seq):
    """
    Estandariza variables secuenciales usando solo el conjunto de entrenamiento.
    train_seq shape: (samples, time_steps, features)
    """
    n_train, t_steps, n_feat = train_seq.shape
    scaler = StandardScaler()

    train_2d = train_seq.reshape(-1, n_feat)
    val_2d = val_seq.reshape(-1, n_feat)
    test_2d = test_seq.reshape(-1, n_feat)

    scaler.fit(train_2d)

    train_scaled = scaler.transform(train_2d).reshape(n_train, t_steps, n_feat)
    val_scaled = scaler.transform(val_2d).reshape(val_seq.shape[0], t_steps, n_feat)
    test_scaled = scaler.transform(test_2d).reshape(test_seq.shape[0], t_steps, n_feat)

    return train_scaled, val_scaled, test_scaled


# =========================
# Construcción de ventanas
# =========================
def create_sequences(np: Any, df, seq_cols, ctx_cols, target_col, time_steps, site_col):
    """
    Crea secuencias por sitio para evitar mezclar tramos distintos.
    Devuelve:
        X_seq: (samples, time_steps, n_seq_features)
        X_ctx: (samples, n_ctx_features_raw)
        y:     (samples,)
    """
    X_seq_list = []
    X_ctx_list = []
    y_list = []

    for _, group in df.groupby(site_col):
        group = group.sort_values("timestamp").reset_index(drop=True)

        seq_values = group[seq_cols].values
        ctx_values = group[ctx_cols].values
        y_values = group[target_col].values

        if len(group) <= time_steps:
            continue

        for i in range(time_steps, len(group)):
            X_seq_list.append(seq_values[i - time_steps : i])
            X_ctx_list.append(ctx_values[i])
            y_list.append(y_values[i])

    if not X_seq_list:
        raise ValueError("No se pudieron generar secuencias. Revisa el tamaño de la ventana temporal.")

    X_seq = np.array(X_seq_list, dtype=np.float32)
    X_ctx = np.array(X_ctx_list, dtype=object)
    y = np.array(y_list, dtype=np.int32)

    return X_seq, X_ctx, y


# =========================
# Modelo híbrido
# =========================
def build_hybrid_lstm(tf: Any, time_steps: int, n_seq_features: int, n_ctx_features: int, n_classes: int = 3):
    """
    Construye el modelo híbrido:
    Rama 1: secuencia temporal con LSTM
    Rama 2: contexto con Dense
    """
    Input = tf.keras.Input
    Model = tf.keras.Model
    LSTM = tf.keras.layers.LSTM
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    Concatenate = tf.keras.layers.Concatenate

    # Entrada secuencial
    seq_input = Input(shape=(time_steps, n_seq_features), name="seq_input")
    x = LSTM(64, return_sequences=True, name="lstm_1")(seq_input)
    x = Dropout(0.20, name="dropout_seq_1")(x)
    x = LSTM(32, return_sequences=False, name="lstm_2")(x)

    # Entrada contextual
    ctx_input = Input(shape=(n_ctx_features,), name="ctx_input")
    c = Dense(16, activation="relu", name="dense_ctx_1")(ctx_input)
    c = Dropout(0.10, name="dropout_ctx_1")(c)

    # Fusión
    merged = Concatenate(name="fusion")([x, c])
    d = Dense(16, activation="relu", name="dense_fusion")(merged)
    output = Dense(n_classes, activation="softmax", name="output")(d)

    model = Model(inputs=[seq_input, ctx_input], outputs=output, name="Hybrid_LSTM_Traffic")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# =========================
# Flujo principal
# =========================
def parse_args() -> argparse.Namespace:
    """Interpreta las opciones de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena el modelo híbrido LSTM con un archivo CSV de tráfico."
    )
    parser.add_argument(
        "--csv",
        default=CFG.csv_path,
        help=f"Ruta del dataset CSV (predeterminado: {CFG.csv_path}).",
    )
    return parser.parse_args()


def main(csv_path: str | None = None) -> None:
    libs = ensure_dependencies()

    np = libs["numpy"]
    pd = libs["pandas"]
    tf = libs["tensorflow"]

    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    selected_csv_path = csv_path or CFG.csv_path
    df = load_data(pd, selected_csv_path)
    print(f"Dataset cargado: {selected_csv_path} ({len(df)} observaciones)")

    # Variables secuenciales
    seq_cols = [
        "flow_total",
        "motos",
        "buses",
        "camiones",
        "velocidad_prom",
        "tiempo_recorrido",
        "cola",
        "incidente",
        "lluvia",
    ]

    # Variables contextuales
    numeric_context_cols = ["carriles", "hora_pico", "feriado"]
    categorical_context_cols = ["tipo_via", "zona"]
    ctx_cols = numeric_context_cols + categorical_context_cols

    # Crear secuencias
    X_seq, X_ctx_raw, y = create_sequences(
        np=np,
        df=df,
        seq_cols=seq_cols,
        ctx_cols=ctx_cols,
        target_col=CFG.target_col,
        time_steps=CFG.time_steps,
        site_col=CFG.site_col,
    )

    # División estratificada
    X_seq_train, X_seq_test, X_ctx_train_raw, X_ctx_test_raw, y_train, y_test = train_test_split(
        X_seq,
        X_ctx_raw,
        y,
        test_size=CFG.test_size,
        random_state=CFG.random_state,
        stratify=y,
    )

    X_seq_train, X_seq_val, X_ctx_train_raw, X_ctx_val_raw, y_train, y_val = train_test_split(
        X_seq_train,
        X_ctx_train_raw,
        y_train,
        test_size=CFG.val_size,
        random_state=CFG.random_state,
        stratify=y_train,
    )

    # Escalado de secuencias
    X_seq_train, X_seq_val, X_seq_test = scale_sequence_features(
        StandardScaler,
        X_seq_train,
        X_seq_val,
        X_seq_test,
    )

    # Preprocesamiento contextual
    preprocessor = build_preprocessor(
        StandardScaler=StandardScaler,
        OneHotEncoder=OneHotEncoder,
        Pipeline=Pipeline,
        ColumnTransformer=ColumnTransformer,
        numeric_context_cols=numeric_context_cols,
        categorical_context_cols=categorical_context_cols,
    )

    X_ctx_train = preprocessor.fit_transform(pd.DataFrame(X_ctx_train_raw, columns=ctx_cols))
    X_ctx_val = preprocessor.transform(pd.DataFrame(X_ctx_val_raw, columns=ctx_cols))
    X_ctx_test = preprocessor.transform(pd.DataFrame(X_ctx_test_raw, columns=ctx_cols))

    # Asegurar arrays densos float32
    if hasattr(X_ctx_train, "toarray"):
        X_ctx_train = X_ctx_train.toarray()
        X_ctx_val = X_ctx_val.toarray()
        X_ctx_test = X_ctx_test.toarray()

    X_ctx_train = X_ctx_train.astype(np.float32)
    X_ctx_val = X_ctx_val.astype(np.float32)
    X_ctx_test = X_ctx_test.astype(np.float32)

    n_seq_features = X_seq_train.shape[2]
    n_ctx_features = X_ctx_train.shape[1]
    n_classes = len(np.unique(y))

    model = build_hybrid_lstm(
        tf=tf,
        time_steps=CFG.time_steps,
        n_seq_features=n_seq_features,
        n_ctx_features=n_ctx_features,
        n_classes=n_classes,
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x={"seq_input": X_seq_train, "ctx_input": X_ctx_train},
        y=y_train,
        validation_data=({"seq_input": X_seq_val, "ctx_input": X_ctx_val}, y_val),
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluación
    test_loss, test_acc = model.evaluate({"seq_input": X_seq_test, "ctx_input": X_ctx_test}, y_test, verbose=0)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Predicciones
    y_proba = model.predict({"seq_input": X_seq_test, "ctx_input": X_ctx_test}, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=4))

    # Guardado
    model.save("hybrid_lstm_traffic_model.keras")
    print("\nModelo guardado en: hybrid_lstm_traffic_model.keras")


if __name__ == "__main__":
    try:
        args = parse_args()
        main(csv_path=args.csv)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"❌ Error de validación: {exc}")
        sys.exit(1)
