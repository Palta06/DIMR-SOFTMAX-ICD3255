"""Etapa 8: Softmax, entropía cruzada media y RMSprop (SOFTMAX pp. 10-15)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from comun import (check_settings, fit_standardizer, load_artifact, load_config,
                   log_stage, preparation_settings, pyplot, rmsprop_step,
                   save_artifact, sha256_file, stage_arguments, standardize,
                   validate_dataset)
from dimr import transform_features


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("Softmax requiere una matriz finita de puntuaciones.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def loss_and_gradient(q: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """E=-sum(T*ln(P))/M; G=Q^T(P-T)/M, sin regularización añadida."""
    logits = q @ weights
    if targets.shape != logits.shape or len(q) == 0:
        raise ValueError("Etiquetas y puntuaciones Softmax incompatibles.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probability = np.exp(log_probability)
    cost = -float(np.sum(targets * log_probability) / len(q))
    gradient = q.T @ (probability - targets) / len(q)
    if not np.isfinite(cost) or not np.isfinite(gradient).all():
        raise FloatingPointError("Costo o gradiente Softmax no finito.")
    return cost, gradient


def design_matrix(x: np.ndarray, intercept: bool) -> np.ndarray:
    return np.column_stack((np.ones(len(x)), x)) if intercept else x


def fit_softmax(q: np.ndarray, targets: np.ndarray, settings: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(settings["seed"])
    weights = rng.normal(0.0, 0.01, (q.shape[1], targets.shape[1]))
    accumulator = np.zeros_like(weights)
    initial, _ = loss_and_gradient(q, targets, weights)
    best_cost, best_epoch = initial, 0
    best_weights, best_accumulator = weights.copy(), accumulator.copy()
    history = [[0, initial]]
    for epoch in range(1, settings["epochs"] + 1):
        permutation = rng.permutation(len(q))
        for start in range(0, len(q), settings["batch_size"]):
            indices = permutation[start:start + settings["batch_size"]]
            _, gradient = loss_and_gradient(q[indices], targets[indices], weights)
            weights, accumulator = rmsprop_step(weights, gradient, accumulator,
                                                settings["learning_rate"], settings["beta"], settings["epsilon"])
        cost, _ = loss_and_gradient(q, targets, weights)
        history.append([epoch, cost])
        if cost < best_cost:
            best_cost, best_epoch = cost, epoch
            best_weights, best_accumulator = weights.copy(), accumulator.copy()
        if epoch == 1 or epoch % 100 == 0 or epoch == settings["epochs"]:
            print(f"Softmax {epoch}/{settings['epochs']}: CE={cost:.8f}", flush=True)
    return {"W": best_weights, "softmax_accumulator": best_accumulator,
            "cost_history": np.asarray(history)}, {
                "selected_epoch": best_epoch, "initial_cost": initial, "selected_cost": best_cost,
                "selection": "minimum_training_cost"}


def train_classifier(cfg: dict, output: Path) -> dict:
    output = Path(output).resolve()
    data, data_meta = load_artifact(output/"dtrain", "dtrain")
    weights, reduction_meta = load_artifact(output/"pesos", "pesos")
    validate_dataset(data, data_meta)
    check_settings(preparation_settings(cfg), data_meta["preparation_settings"], "Preparación")
    check_settings({"sls": cfg["sls"], "pca": cfg["pca"]}, reduction_meta["settings"], "Reducción")
    if reduction_meta["data_run_id"] != data_meta["data_run_id"] or reduction_meta["dtrain_sha256"] != sha256_file(output/"dtrain"):
        raise ValueError("pesos no corresponde al dtrain actual; ejecute dimr.py.")
    reduced = transform_features(data["X"], weights)
    mean, scale = fit_standardizer(reduced, cfg["normalization"]["ddof"])
    q = design_matrix(standardize(reduced, mean, scale), cfg["softmax"]["intercept"])
    arrays, report = fit_softmax(q, data["T"], cfg["softmax"])
    arrays.update({"reduced_mean": mean, "reduced_scale": scale})
    metadata = {"data_run_id": data_meta["data_run_id"], "class_names": data_meta["class_names"],
                "pesos_sha256": sha256_file(output/"pesos"), "settings": cfg["softmax"],
                "normalization": cfg["normalization"], "training": report}
    save_artifact(output/"pesoftmax", "pesoftmax", metadata, **arrays)
    np.savetxt(output/"costo", arrays["cost_history"], delimiter=",", fmt="%.17g",
               header="epoch,cross_entropy", comments="")
    result_dir = output/"Resultados"
    result_dir.mkdir(exist_ok=True)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(arrays["cost_history"][:, 0], arrays["cost_history"][:, 1], color="#176B87")
    ax.scatter([report["selected_epoch"]], [report["selected_cost"]], color="#BA4A00", zorder=3,
               label=f"Modelo guardado: época {report['selected_epoch']}")
    ax.set(xlabel="Época", ylabel="Entropía cruzada media (logaritmo natural)",
           title="Softmax + RMSprop · entrenamiento HP0–HP2")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(result_dir/"grafica_costo.png", dpi=160)
    plt.close(fig)
    complete = {**metadata, "pesoftmax_sha256": sha256_file(output/"pesoftmax")}
    log_stage(output, "train", complete)
    print(f"pesoftmax guardado: época {report['selected_epoch']}, CE={report['selected_cost']:.8f}.", flush=True)
    return complete


def main() -> None:
    args = stage_arguments("Entrenar Softmax en dtrain; guardar pesoftmax y costo.")
    cfg, _ = load_config(args.config)
    train_classifier(cfg, args.output_dir)


if __name__ == "__main__":
    main()
