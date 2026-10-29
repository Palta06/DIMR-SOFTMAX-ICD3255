"""Etapas 3-5: DE/MDE, ventanas, partición por carga y normalización.

Fuentes: DISP_Entropy pp. 2-4; MultEnt pp. 4-5; Enunciado p. 5.
Las cargas HP0..HP3 son columnas independientes, nunca canales concatenados.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from comun import (array_digest, fingerprint, fit_standardizer, load_config, log_stage,
                   preparation_settings, save_artifact, save_json, sha256_file,
                   stage_arguments, standardize, positive_int)


def dispersion_symbols(x: np.ndarray, symbols: int) -> np.ndarray:
    """Min-max y round(c*x+0.5), con símbolos en 1..c (ejemplos del PDF).

    Para valores no negativos, redondear mitades hacia arriba equivale a
    floor(c*x+1). El máximo se limita a c. No usar np.round (empates al par).
    Una serie constante recibe un solo símbolo y tiene entropía cero.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or symbols < 2:
        raise ValueError("Serie o número de símbolos inválido.")
    width = np.ptp(x)
    if width == 0:
        return np.ones(len(x), dtype=np.int64)
    normalized = (x - x.min()) / width
    return np.clip(np.floor(symbols * normalized + 1.0), 1, symbols).astype(np.int64)


def dispersion_entropy(x: np.ndarray, dimension: int, delay: int,
                       symbols: int, normalized: bool = True) -> float:
    for name, value in (("dimension", dimension), ("delay", delay), ("symbols", symbols)):
        positive_int(value, name)
    x = np.asarray(x, dtype=np.float64)
    count = len(x) - (dimension - 1) * delay
    if count < 1:
        raise ValueError("La serie es demasiado corta para el embedding solicitado.")
    patterns = symbols**dimension
    if patterns > 10_000_000:
        raise ValueError("c^d excede el límite de memoria para el conteo explícito.")
    mapped = dispersion_symbols(x, symbols)
    indices = np.arange(count)[:, None] + delay * np.arange(dimension)[None, :]
    embedded = mapped[indices]
    # PDF: k=1+dot(Y_i-1,c**arange(d)). NumPy utiliza índices desde cero.
    codes = (embedded - 1) @ (symbols ** np.arange(dimension, dtype=np.int64))
    frequency = np.bincount(codes, minlength=patterns)
    probability = frequency[frequency > 0] / count
    # 0*log2(0)=0: solo se suman probabilidades positivas.
    entropy = -float(np.sum(probability * np.log2(probability)))
    return entropy / np.log2(patterns) if normalized else entropy


def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    positive_int(scale, "scale")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("La serie debe ser un vector.")
    count = len(x) // scale
    if count == 0:
        raise ValueError("La escala excede la longitud de la serie.")
    return x[:count * scale].reshape(count, scale).mean(axis=1)


def multiscale_dispersion_entropy(x: np.ndarray, dimension: int, delay: int,
                                 symbols: int, max_scale: int,
                                 normalized: bool = True) -> np.ndarray:
    positive_int(max_scale, "max_scale")
    # Cada escala se calcula desde la misma ventana original.
    return np.array([dispersion_entropy(coarse_grain(x, scale), dimension, delay,
                                       symbols, normalized)
                     for scale in range(1, max_scale + 1)])


def prepare_data(cfg: dict, config_path: Path, output: Path) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (config_path.parent / cfg["data"]["directory"]).resolve()
    w, step = cfg["window"]["length"], cfg["window"]["step"]
    train_columns = set(cfg["split"]["train_columns"])
    features, labels, provenance, train_mask = [], [], [], []
    inventory = []
    for class_id, info in enumerate(cfg["data"]["classes"]):
        path = source / info["file"]
        signal = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        expected = (cfg["data"]["expected_rows"], len(cfg["data"]["columns"]))
        if signal.shape != expected or not np.isfinite(signal).all():
            raise ValueError(f"{path.name}: se esperaba {expected} con valores finitos; recibido {signal.shape}.")
        starts = range(0, len(signal) - w + 1, step)
        if not len(starts):
            raise ValueError(f"{path.name}: no cabe ninguna ventana completa.")
        inventory.append({"file": path.name, "class_id": class_id,
                          "class_name": info["name"], "shape": list(signal.shape),
                          "sha256": sha256_file(path), "windows_per_column": len(starts),
                          "discarded_tail_samples": len(signal) - (starts[-1] + w)})
        for column, column_name in enumerate(cfg["data"]["columns"]):
            for start in starts:
                features.append(multiscale_dispersion_entropy(signal[start:start+w, column], **cfg["mde"]))
                labels.append(class_id)
                provenance.append((class_id, column, start, start + w))
                train_mask.append(column in train_columns)
        print(f"MDE: {path.name}, {len(starts)} ventanas por columna.", flush=True)
    f = np.asarray(features)
    class_ids = np.asarray(labels, dtype=np.int64)
    origins = np.asarray(provenance, dtype=np.int64)
    mask = np.asarray(train_mask, dtype=bool)
    if not np.isfinite(f).all():
        raise ValueError("Hay características MDE no finitas.")
    # Una misma permutación para features, etiquetas, procedencia y partición.
    order = np.random.default_rng(cfg["seed"]).permutation(len(f))
    f, class_ids, origins, mask = f[order], class_ids[order], origins[order], mask[order]
    t = np.eye(4, dtype=np.int64)[class_ids]
    mean, scale = fit_standardizer(f[mask], cfg["normalization"]["ddof"])
    x = standardize(f, mean, scale)
    metadata = {"preparation_settings": preparation_settings(cfg),
                "class_names": [c["name"] for c in cfg["data"]["classes"]],
                "column_names": cfg["data"]["columns"],
                "input_is_standardized": True, "inventory": inventory,
                "origin_columns": ["class_id", "column_index", "start_inclusive", "stop_exclusive"]}
    # Identificar también el contenido calculado: cambiar la extracción debe
    # invalidar los modelos previos aunque se conserve la configuración.
    metadata["data_run_id"] = fingerprint({"metadata": metadata,
        "prepared_arrays_sha256": array_digest({"X": x, "T": t, "origins": origins,
                                                 "train_mask": mask, "mean": mean, "scale": scale})})
    for name, selection in (("dtrain", mask), ("dtest", ~mask)):
        if set(class_ids[selection]) != set(range(4)):
            raise ValueError(f"{name} no contiene las cuatro clases.")
        save_artifact(output/name, name, metadata,
                      X=x[selection], T=t[selection], F_raw=f[selection],
                      origins=origins[selection], global_row=np.flatnonzero(selection),
                      input_mean=mean, input_scale=scale)
    np.savetxt(output/"dfeatures.csv", x, delimiter=",", fmt="%.17g")
    np.savetxt(output/"dclass.csv", t, delimiter=",", fmt="%d")
    provenance_path = output.parent/"Trabajo"/"registros"/"procedencia.csv"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "file", "class_id", "class_name", "load", "start", "stop", "split"])
        for row, ((cid, col, start, stop), is_train) in enumerate(zip(origins, mask)):
            writer.writerow([row, inventory[cid]["file"], cid, metadata["class_names"][cid],
                             cfg["data"]["columns"][col], start, stop, "train" if is_train else "test"])
    report = {"data_run_id": metadata["data_run_id"], "n_train": int(mask.sum()),
              "n_test": int((~mask).sum()), "features": x.shape[1],
              "train_fraction": float(mask.mean()), "inventory": inventory,
              "train_counts": np.bincount(class_ids[mask], minlength=4).tolist(),
              "test_counts": np.bincount(class_ids[~mask], minlength=4).tolist(),
              "preparation_settings": preparation_settings(cfg)}
    log_stage(output, "mdef", report)
    print(f"Preparación: {report['n_train']} training, {report['n_test']} testing, D={x.shape[1]}.", flush=True)
    return report


def main() -> None:
    args = stage_arguments("Extraer MDE y crear dfeatures.csv, dclass.csv, dtrain y dtest.")
    cfg, path = load_config(args.config)
    prepare_data(cfg, path, args.output_dir)


if __name__ == "__main__":
    main()
