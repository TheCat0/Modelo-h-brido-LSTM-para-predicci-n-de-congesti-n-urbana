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
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List


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
        "joblib": "joblib",
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


def calculate_partition_sizes(
    total_rows: int,
    time_steps: int,
    test_size: float,
    val_size: float,
) -> tuple[int, int, int, int]:
    """Calcula tamaños cronológicos con dos brechas purgadas entre particiones."""
    gap = max(0, time_steps - 1)
    minimum_partition_rows = time_steps + 1
    usable_rows = total_rows - 2 * gap
    minimum_required = 3 * minimum_partition_rows
    if usable_rows < minimum_required:
        raise ValueError(
            f"Se requieren al menos {minimum_required + 2 * gap} filas para "
            f"train/validación/test con time_steps={time_steps} y brechas de {gap} filas."
        )

    test_rows = max(minimum_partition_rows, round(usable_rows * test_size))
    remaining_after_test = usable_rows - test_rows
    val_rows = max(minimum_partition_rows, round(remaining_after_test * val_size))
    train_rows = usable_rows - val_rows - test_rows

    if train_rows < minimum_partition_rows:
        deficit = minimum_partition_rows - train_rows
        reducible_val = max(0, val_rows - minimum_partition_rows)
        shift = min(deficit, reducible_val)
        val_rows -= shift
        train_rows += shift
        deficit -= shift
        if deficit:
            test_rows -= deficit
            train_rows += deficit

    if min(train_rows, val_rows, test_rows) < minimum_partition_rows:
        raise ValueError("No fue posible asignar suficientes filas a todas las particiones.")

    return train_rows, val_rows, test_rows, gap


def split_data_chronologically(
    pd: Any,
    df,
    site_col: str,
    timestamp_col: str,
    time_steps: int,
    test_size: float,
    val_size: float,
):
    """Divide cada sitio por tiempo y purga los límites para evitar ventanas solapadas."""
    train_groups = []
    val_groups = []
    test_groups = []
    split_metadata = {}

    for site_id, group in df.groupby(site_col, sort=False):
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        try:
            train_rows, val_rows, test_rows, gap = calculate_partition_sizes(
                total_rows=len(group),
                time_steps=time_steps,
                test_size=test_size,
                val_size=val_size,
            )
        except ValueError as exc:
            raise ValueError(f"El sitio {site_id!r}: {exc}") from exc

        train_end = train_rows
        val_start = train_end + gap
        val_end = val_start + val_rows
        test_start = val_end + gap

        train_group = group.iloc[:train_end].copy()
        val_group = group.iloc[val_start:val_end].copy()
        test_group = group.iloc[test_start : test_start + test_rows].copy()

        train_groups.append(train_group)
        val_groups.append(val_group)
        test_groups.append(test_group)
        split_metadata[str(site_id)] = {
            "source_rows": len(group),
            "gap_rows": gap,
            "train_rows": len(train_group),
            "validation_rows": len(val_group),
            "test_rows": len(test_group),
            "train_range": [
                train_group[timestamp_col].iloc[0].isoformat(),
                train_group[timestamp_col].iloc[-1].isoformat(),
            ],
            "validation_range": [
                val_group[timestamp_col].iloc[0].isoformat(),
                val_group[timestamp_col].iloc[-1].isoformat(),
            ],
            "test_range": [
                test_group[timestamp_col].iloc[0].isoformat(),
                test_group[timestamp_col].iloc[-1].isoformat(),
            ],
        }

    return (
        pd.concat(train_groups, ignore_index=True),
        pd.concat(val_groups, ignore_index=True),
        pd.concat(test_groups, ignore_index=True),
        split_metadata,
    )


def fit_and_scale_sequence_features(StandardScaler: Any, train_df, val_df, test_df, seq_cols):
    """Ajusta el escalador solo con filas de entrenamiento y transforma cada partición."""
    scaler = StandardScaler()
    scaler.fit(train_df[seq_cols])

    scaled_partitions = []
    for partition in (train_df, val_df, test_df):
        scaled = partition.copy()
        scaled.loc[:, seq_cols] = scaler.transform(partition[seq_cols])
        scaled_partitions.append(scaled)

    return scaler, tuple(scaled_partitions)


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


def save_artifacts(
    joblib: Any,
    model,
    sequence_scaler,
    context_preprocessor,
    metadata: dict[str, Any],
    artifacts_dir: str,
) -> None:
    """Guarda el modelo, los preprocesadores ajustados y metadatos de inferencia."""
    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "hybrid_lstm_traffic_model.keras"
    sequence_scaler_path = output_dir / "sequence_scaler.joblib"
    context_preprocessor_path = output_dir / "context_preprocessor.joblib"
    metadata_path = output_dir / "model_metadata.json"

    model.save(model_path)
    joblib.dump(sequence_scaler, sequence_scaler_path)
    joblib.dump(context_preprocessor, context_preprocessor_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nArtefactos guardados en: {output_dir}")
    for artifact in (model_path, sequence_scaler_path, context_preprocessor_path, metadata_path):
        print(f"  - {artifact}")


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
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directorio de salida para modelo, preprocesadores y metadatos.",
    )
    return parser.parse_args()


def main(csv_path: str | None = None, artifacts_dir: str = "artifacts") -> None:
    libs = ensure_dependencies()

    np = libs["numpy"]
    pd = libs["pandas"]
    tf = libs["tensorflow"]
    sklearn = libs["sklearn"]
    joblib = libs["joblib"]

    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import classification_report, confusion_matrix
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

    # División cronológica por sitio antes de construir ventanas. Las brechas
    # impiden que train, validación y test compartan timesteps.
    train_df, val_df, test_df, split_metadata = split_data_chronologically(
        pd=pd,
        df=df,
        site_col=CFG.site_col,
        timestamp_col=CFG.timestamp_col,
        time_steps=CFG.time_steps,
        test_size=CFG.test_size,
        val_size=CFG.val_size,
    )

    sequence_scaler, scaled_partitions = fit_and_scale_sequence_features(
        StandardScaler=StandardScaler,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        seq_cols=seq_cols,
    )
    train_scaled_df, val_scaled_df, test_scaled_df = scaled_partitions

    X_seq_train, X_ctx_train_raw, y_train = create_sequences(
        np, train_scaled_df, seq_cols, ctx_cols, CFG.target_col, CFG.time_steps, CFG.site_col
    )
    X_seq_val, X_ctx_val_raw, y_val = create_sequences(
        np, val_scaled_df, seq_cols, ctx_cols, CFG.target_col, CFG.time_steps, CFG.site_col
    )
    X_seq_test, X_ctx_test_raw, y_test = create_sequences(
        np, test_scaled_df, seq_cols, ctx_cols, CFG.target_col, CFG.time_steps, CFG.site_col
    )

    # El preprocesador contextual también se ajusta exclusivamente con train.
    preprocessor = build_preprocessor(
        StandardScaler=StandardScaler,
        OneHotEncoder=OneHotEncoder,
        Pipeline=Pipeline,
        ColumnTransformer=ColumnTransformer,
        numeric_context_cols=numeric_context_cols,
        categorical_context_cols=categorical_context_cols,
    )
    preprocessor.fit(train_df[ctx_cols])
    X_ctx_train = preprocessor.transform(pd.DataFrame(X_ctx_train_raw, columns=ctx_cols))
    X_ctx_val = preprocessor.transform(pd.DataFrame(X_ctx_val_raw, columns=ctx_cols))
    X_ctx_test = preprocessor.transform(pd.DataFrame(X_ctx_test_raw, columns=ctx_cols))

    if hasattr(X_ctx_train, "toarray"):
        X_ctx_train = X_ctx_train.toarray()
        X_ctx_val = X_ctx_val.toarray()
        X_ctx_test = X_ctx_test.toarray()

    X_ctx_train = X_ctx_train.astype(np.float32)
    X_ctx_val = X_ctx_val.astype(np.float32)
    X_ctx_test = X_ctx_test.astype(np.float32)

    n_seq_features = X_seq_train.shape[2]
    n_ctx_features = X_ctx_train.shape[1]
    class_labels = sorted(int(label) for label in df[CFG.target_col].unique())
    expected_labels = list(range(len(class_labels)))
    if class_labels != expected_labels:
        raise ValueError(
            f"Las clases deben ser enteros contiguos desde 0; se encontraron {class_labels}."
        )
    n_classes = len(class_labels)

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

    categorical_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": selected_csv_path,
        "config": asdict(CFG),
        "model_name": model.name,
        "model_config": json.loads(model.to_json()),
        "sequence_length": CFG.time_steps,
        "sequence_columns": seq_cols,
        "numeric_context_columns": numeric_context_cols,
        "categorical_context_columns": categorical_context_cols,
        "context_columns": ctx_cols,
        "context_feature_names": preprocessor.get_feature_names_out().tolist(),
        "categorical_categories": {
            column: [str(value) for value in categories]
            for column, categories in zip(categorical_context_cols, categorical_encoder.categories_)
        },
        "target_column": CFG.target_col,
        "class_labels": class_labels,
        "split_strategy": "chronological_per_site_with_purged_gaps",
        "split_metadata": split_metadata,
        "window_counts": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
        },
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_artifacts(
        joblib=joblib,
        model=model,
        sequence_scaler=sequence_scaler,
        context_preprocessor=preprocessor,
        metadata=metadata,
        artifacts_dir=artifacts_dir,
    )


if __name__ == "__main__":
    try:
        args = parse_args()
        main(csv_path=args.csv, artifacts_dir=args.artifacts_dir)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"❌ Error de validación: {exc}")
        sys.exit(1)
