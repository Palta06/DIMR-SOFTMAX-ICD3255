"""Etapa 9: inferencia independiente y métricas. Este archivo no ajusta modelos.

SOFTMAX p. 16: matriz con FILAS predichas y COLUMNAS reales.
"""
from __future__ import annotations

import csv
import platform
from pathlib import Path

import numpy as np

from comun import (BASE, check_settings, load_artifact, load_config, log_stage,
                   preparation_settings, pyplot, save_json, sha256_file,
                   stage_arguments, standardize, validate_dataset)
from dimr import transform_features
from train import design_matrix, softmax_probabilities


def confusion_matrix(predicted: np.ndarray, real: np.ndarray, n_classes: int = 4) -> np.ndarray:
    predicted, real = np.asarray(predicted), np.asarray(real)
    if predicted.shape != real.shape or predicted.ndim != 1 or not len(real):
        raise ValueError("Vectores de etiquetas incompatibles o vacíos.")
    if not np.issubdtype(predicted.dtype, np.integer) or not np.issubdtype(real.dtype, np.integer):
        raise ValueError("Las etiquetas deben ser enteras.")
    if np.any(predicted < 0) or np.any(real < 0) or np.any(predicted >= n_classes) or np.any(real >= n_classes):
        raise ValueError("Etiqueta fuera del diccionario de clases.")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (predicted, real), 1)
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0) or matrix.sum() == 0:
        raise ValueError("Matriz de confusión inválida.")
    tp = np.diag(matrix).astype(float)
    predicted_count, support = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=precision + recall != 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support,
            "tp": tp.astype(int), "fp": predicted_count - tp.astype(int), "fn": support - tp.astype(int),
            "accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.dot(f1, support) / support.sum())}


def predict(x: np.ndarray, reduction: dict, classifier: dict, intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    reduced = transform_features(x, reduction)
    normalized = standardize(reduced, classifier["reduced_mean"], classifier["reduced_scale"])
    probabilities = softmax_probabilities(design_matrix(normalized, intercept) @ classifier["W"])
    # Los empates se resuelven por el primer índice (orden fijo class1..class4).
    return probabilities.argmax(axis=1), probabilities


def evaluate(cfg: dict, output: Path) -> dict:
    output = Path(output).resolve()
    data, data_meta = load_artifact(output/"dtest", "dtest")
    reduction, reduction_meta = load_artifact(output/"pesos", "pesos")
    classifier, classifier_meta = load_artifact(output/"pesoftmax", "pesoftmax")
    validate_dataset(data, data_meta)
    check_settings(preparation_settings(cfg), data_meta["preparation_settings"], "Preparación")
    check_settings({"sls": cfg["sls"], "pca": cfg["pca"]}, reduction_meta["settings"], "Reducción")
    check_settings(cfg["softmax"], classifier_meta["settings"], "Softmax")
    check_settings(cfg["normalization"], classifier_meta["normalization"], "Normalización")
    if len({data_meta["data_run_id"], reduction_meta["data_run_id"], classifier_meta["data_run_id"]}) != 1:
        raise ValueError("dtest, pesos y pesoftmax provienen de preparaciones diferentes.")
    if classifier_meta["pesos_sha256"] != sha256_file(output/"pesos"):
        raise ValueError("pesoftmax no corresponde al archivo pesos actual.")
    for key in ("input_mean", "input_scale"):
        if not np.array_equal(data[key], reduction[key]):
            raise ValueError("dtest y pesos no utilizan la misma normalización de entrada.")
    for model_meta in (reduction_meta, classifier_meta):
        check_settings(data_meta["class_names"], model_meta["class_names"], "Diccionario de clases")
    predicted, probability = predict(data["X"], reduction, classifier, cfg["softmax"]["intercept"])
    real = data["T"].argmax(axis=1)
    matrix = confusion_matrix(predicted, real)
    metrics = classification_metrics(matrix)
    results = output/"Resultados"
    results.mkdir(exist_ok=True)
    names = data_meta["class_names"]
    with (results/"matriz_confusion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Predicho / Real"] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + row.tolist())
    with (results/"fscores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_number", "class_name", "precision", "recall", "f1", "support", "tp", "fp", "fn"])
        for k, name in enumerate(names):
            writer.writerow([k + 1, name] + [metrics[key][k] for key in ("precision", "recall", "f1", "support", "tp", "fp", "fn")])
    with (results/"predicciones.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["global_row", "load", "start", "stop", "real_class", "predicted_class",
                         "p_class1", "p_class2", "p_class3", "p_class4"])
        for row, origin, truth, pred, prob in zip(data["global_row"], data["origins"], real, predicted, probability):
            writer.writerow([row, data_meta["column_names"][origin[1]], origin[2], origin[3], truth + 1, pred + 1, *prob])
    summary = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}
    summary.update({"class_names": names, "n_test": len(real), "confusion_matrix": matrix.tolist(),
                    "confusion_orientation": "rows_predicted_columns_real", "zero_division": 0,
                    "data_run_id": data_meta["data_run_id"], "test_loads": [cfg["data"]["columns"][c] for c in cfg["split"]["test_columns"]],
                    "pesos_sha256": sha256_file(output/"pesos"), "pesoftmax_sha256": sha256_file(output/"pesoftmax")})
    save_json(results/"metricas.json", summary)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    picture = ax.imshow(matrix, cmap="Blues", vmin=0)
    tick_names = [name.replace("Daño ", "Daño\n") for name in names]
    ax.set_xticks(range(4), tick_names, fontsize=9)
    ax.set_yticks(range(4), tick_names, fontsize=9)
    ax.set(xlabel="Clase real (columnas)", ylabel="Clase predicha (filas)",
           title=f"Prueba {', '.join(summary['test_loads'])} · {len(real)} ventanas\n"
                 f"Exactitud {metrics['accuracy']:.2%} · F1 macro {metrics['macro_f1']:.4f}")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(picture, ax=ax, label="Número de ventanas", shrink=.8)
    fig.savefig(results/"matriz_confusion.png", dpi=160, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    tracked = [output / name for name in ("dfeatures.csv", "dclass.csv", "dtrain", "dtest", "pesos", "pesoftmax", "costo")]
    tracked += sorted(p for p in results.iterdir() if p.is_file() and p.name not in ("ejecucion.json", "validacion_cv.csv"))
    manifest = {"configuracion": cfg, "data_run_id": data_meta["data_run_id"],
                "python_version": platform.python_version(), "numpy_version": np.__version__,
                "files_sha256": {p.relative_to(output).as_posix(): sha256_file(p) for p in tracked if p.exists()},
                "code_sha256": {name: sha256_file(BASE/name) for name in ("comun.py", "mdef.py", "dimr.py", "train.py", "test.py")}}
    save_json(results/"ejecucion.json", manifest)
    log_stage(output, "test", summary)
    print(f"Prueba: accuracy={metrics['accuracy']:.4%}, F1 macro={metrics['macro_f1']:.6f}, N={len(real)}.", flush=True)
    return summary


def main() -> None:
    args = stage_arguments("Evaluar dtest usando exclusivamente pesos y pesoftmax guardados.")
    cfg, _ = load_config(args.config)
    evaluate(cfg, args.output_dir)


if __name__ == "__main__":
    main()
