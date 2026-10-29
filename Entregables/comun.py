"""Persistencia, normalización y utilidades compartidas. Sin efectos al importar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        if a.dtype.hasobject:
            raise ValueError("No se permiten arrays de objetos ni pickle.")
        digest.update(json_text([name, a.dtype.str, list(a.shape)]).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def save_artifact(path: Path, kind: str, metadata: dict, **arrays: np.ndarray) -> None:
    """NPZ seguro conservando el nombre EXACTO, sin añadir .npz."""
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    if "metadata" in payload:
        raise ValueError("metadata es una clave reservada.")
    meta = {**metadata, "kind": kind, "schema_version": SCHEMA_VERSION,
            "array_sha256": array_digest(payload)}
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json_text(meta)), **payload)
    temporary.replace(path)


def load_artifact(path: Path, kind: str) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path.name}. Ejecute la etapa previa.")
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
    if meta.get("kind") != kind or meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Formato incompatible de {path.name}.")
    if meta.get("array_sha256") != array_digest(arrays):
        raise ValueError(f"La integridad de {path.name} no coincide con su registro.")
    return arrays, meta


def preparation_settings(cfg: dict) -> dict:
    return {key: cfg[key] for key in ("seed", "data", "window", "mde", "split", "normalization")}


def check_settings(actual: Any, expected: Any, label: str) -> None:
    if fingerprint(actual) != fingerprint(expected):
        raise ValueError(f"{label}: configuración o procedencia incompatible; regenere etapas previas.")


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo.")
    return value


def load_config(path: Path | None = None) -> tuple[dict, Path]:
    path = Path(path).resolve() if path else BASE / "configuracion.json"
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for section in ("data", "window", "mde", "split", "normalization", "sls", "pca", "softmax"):
        if section not in cfg:
            raise ValueError(f"Falta la sección {section} en {path.name}.")
    window = positive_int(cfg["window"]["length"], "window.length")
    positive_int(cfg["window"]["step"], "window.step")
    for key in ("dimension", "delay", "symbols", "max_scale"):
        positive_int(cfg["mde"][key], "mde." + key)
    if cfg["mde"]["symbols"] < 2:
        raise ValueError("MDE requiere al menos dos símbolos.")
    if window // cfg["mde"]["max_scale"] - (cfg["mde"]["dimension"] - 1) * cfg["mde"]["delay"] < 1:
        raise ValueError("La escala máxima no permite formar un patrón de embedding.")
    if not isinstance(cfg["mde"]["normalized"], bool):
        raise ValueError("mde.normalized debe ser booleano.")
    classes = cfg["data"]["classes"]
    if len(classes) != 4 or len({c["file"] for c in classes}) != 4:
        raise ValueError("Se requieren cuatro clases y cuatro archivos distintos.")
    train_cols, test_cols = cfg["split"]["train_columns"], cfg["split"]["test_columns"]
    columns = train_cols + test_cols
    if not train_cols or not test_cols or len(set(columns)) != len(columns):
        raise ValueError("Las columnas de training y testing deben ser no vacías y disjuntas.")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in columns):
        raise ValueError("Los índices de columna deben ser enteros.")
    if set(columns) != set(range(len(cfg["data"]["columns"]))):
        raise ValueError("Asigne cada columna una vez a training o testing.")
    if cfg["normalization"]["ddof"] not in (0, 1):
        raise ValueError("ddof debe ser 0 o 1.")
    if cfg["sls"]["activation"] != "cos":
        raise ValueError("Esta implementación aplica f(z)=cos(z), EJERCICIOS pp. 5-6.")
    j = positive_int(cfg["sls"]["latent_dim"], "sls.latent_dim")
    if j >= cfg["mde"]["max_scale"]:
        raise ValueError("La reducción SLS requiere J menor que D=max_scale.")
    if cfg["sls"]["lambda"] <= 0:
        raise ValueError("Use lambda > 0 para una solución RLS única.")
    for section in ("sls", "softmax"):
        block = cfg[section]
        for key in ("batch_size", "epochs"):
            positive_int(block[key], section + "." + key)
        if not 0 < block["learning_rate"] < 1 or not 0 <= block["beta"] < 1 or block["epsilon"] <= 0:
            raise ValueError(f"Parámetros RMSprop inválidos en {section}.")
    if not 0 < cfg["pca"]["variance_threshold"] <= 1:
        raise ValueError("El umbral PCA debe estar en (0,1].")
    if cfg["pca"]["criterion"] not in ("at_most", "at_least"):
        raise ValueError("Criterio PCA inválido.")
    if not isinstance(cfg["softmax"]["intercept"], bool):
        raise ValueError("softmax.intercept debe ser booleano.")
    return cfg, path


def stage_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Configuración alternativa; por defecto junto al script.")
    parser.add_argument("--output-dir", type=Path, default=BASE,
                        help="Carpeta de artefactos; por defecto Entregables.")
    return parser.parse_args()


def fit_standardizer(x: np.ndarray, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or len(x) <= ddof or not np.isfinite(x).all():
        raise ValueError("Datos insuficientes o no finitos para normalizar.")
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=ddof)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or mean.shape != (x.shape[1],) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("Dimensiones o divisores de normalización incompatibles.")
    result = (x - mean) / scale
    if not np.isfinite(result).all():
        raise ValueError("La normalización produjo valores no finitos.")
    return result


def rmsprop_step(theta: np.ndarray, gradient: np.ndarray, accumulator: np.ndarray,
                 rate: float, beta: float, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """SOFTMAX p. 15: epsilon FUERA de la raíz; operaciones elemento a elemento."""
    if theta.shape != gradient.shape or theta.shape != accumulator.shape:
        raise ValueError("Formas incompatibles en RMSprop.")
    accumulator = beta * accumulator + (1 - beta) * gradient**2
    theta = theta - rate * gradient / (np.sqrt(accumulator) + epsilon)
    if not np.isfinite(theta).all() or not np.isfinite(accumulator).all():
        raise FloatingPointError("RMSprop produjo valores no finitos.")
    return theta, accumulator


def validate_dataset(data: dict, meta: dict) -> None:
    x, t = data["X"], data["T"]
    if x.ndim != 2 or len(x) == 0 or t.shape != (len(x), 4):
        raise ValueError("Dimensiones inválidas de features o etiquetas.")
    if not np.isfinite(x).all() or not np.all((t == 0) | (t == 1)) or not np.all(t.sum(axis=1) == 1):
        raise ValueError("Features no finitas o etiquetas que no son one-hot.")
    if len(meta["class_names"]) != 4:
        raise ValueError("Diccionario de clases inválido.")


def log_stage(output: Path, stage: str, record: dict) -> None:
    root = Path(output).resolve().parent
    save_json(root / "Trabajo" / "registros" / f"{stage}.json", record)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
