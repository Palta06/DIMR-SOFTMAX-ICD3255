"""Etapas 6-7: SLS coseno/RLS/RMSprop y PCA por SVD.

EJERCICIOS pp. 2-5 fija el costo y la activación. Enunciado p. 6 exige
RMSprop para V. Integral-PCA pp. 3-12 define centrado, SVD y energía.
Todas las observaciones son FILAS: X(N,D), V(D,J), H(N,J), U(J,D).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from comun import (check_settings, fingerprint, load_artifact, load_config,
                   log_stage, preparation_settings, pyplot, rmsprop_step,
                   save_artifact, sha256_file, stage_arguments, standardize,
                   validate_dataset)


def solve_rls(h: np.ndarray, x: np.ndarray, regularization: float) -> np.ndarray:
    """U=(H^T H + lambda I)^(-1) H^T X; solve evita formar la inversa."""
    if h.ndim != 2 or x.ndim != 2 or len(h) != len(x) or regularization <= 0:
        raise ValueError("Dimensiones RLS o regularización inválidas.")
    return np.linalg.solve(h.T @ h + regularization * np.eye(h.shape[1]), h.T @ x)


def sls_cost(x: np.ndarray, v: np.ndarray, u: np.ndarray, regularization: float) -> float:
    residual = np.cos(x @ v) @ u - x
    return float(0.5 * np.sum(residual**2) + 0.5 * regularization * np.sum(u**2))


def sls_gradient(x: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Gradiente del costo SUMADO, U fijo. f'(Z)=-sin(Z), sin factor 1/N.

    Se adapta el orden matricial de la lámina a sus dimensiones declaradas;
    la expresión resultante tiene la misma forma D x J de V.
    """
    z = x @ v
    error = np.cos(z) @ u - x
    return -x.T @ ((error @ u.T) * np.sin(z))


def fit_sls(x: np.ndarray, settings: dict) -> tuple[dict, dict]:
    rng = np.random.default_rng(settings["seed"])
    v = rng.normal(0.0, 1 / np.sqrt(x.shape[1]), (x.shape[1], settings["latent_dim"]))
    accumulator = np.zeros_like(v)
    lam = settings["lambda"]
    u = solve_rls(np.cos(x @ v), x, lam)
    initial = sls_cost(x, v, u, lam)
    best_cost, best_epoch = initial, 0
    best_v, best_u, best_accumulator = v.copy(), u.copy(), accumulator.copy()
    history = [[0, initial, float(np.mean((np.cos(x @ v) @ u - x)**2))]]
    batch_size = settings["batch_size"]
    for epoch in range(1, settings["epochs"] + 1):
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = x[permutation[start:start + batch_size]]
            h = np.cos(batch @ v)
            u_batch = solve_rls(h, batch, lam)
            gradient = sls_gradient(batch, v, u_batch)
            v, accumulator = rmsprop_step(v, gradient, accumulator,
                                          settings["learning_rate"], settings["beta"], settings["epsilon"])
        # U definitivo se ajusta a TODO training para el V de esta época.
        h = np.cos(x @ v)
        u = solve_rls(h, x, lam)
        cost = sls_cost(x, v, u, lam)
        mse = float(np.mean((h @ u - x)**2))
        if not np.isfinite(cost):
            raise FloatingPointError("Costo SLS no finito.")
        history.append([epoch, cost, mse])
        if cost < best_cost:
            best_cost, best_epoch = cost, epoch
            best_v, best_u, best_accumulator = v.copy(), u.copy(), accumulator.copy()
        if epoch == 1 or epoch % 50 == 0 or epoch == settings["epochs"]:
            print(f"SLS {epoch}/{settings['epochs']}: E={cost:.6f}, MSE={mse:.6f}", flush=True)
    arrays = {"V": best_v, "U": best_u, "sls_accumulator": best_accumulator,
              "sls_history": np.asarray(history)}
    report = {"selected_epoch": best_epoch, "initial_cost": initial,
              "selected_cost": best_cost, "selection": "minimum_training_cost",
              "reconstruction_mse": float(np.mean((np.cos(x @ best_v) @ best_u - x)**2))}
    return arrays, report


def select_components(ratios: np.ndarray, threshold: float, criterion: str) -> int:
    cumulative = np.cumsum(ratios)
    if criterion == "at_most":
        # Criterio literal del ejemplo de Integral-PCA p. 11 (<= 95%).
        candidates = np.flatnonzero(cumulative <= threshold + 1e-12)
        if not len(candidates):
            raise ValueError("El criterio PCA <= umbral no conserva ningún componente. "
                             "Revise el umbral o elija explícitamente criterion='at_least'.")
        return int(candidates[-1] + 1)
    if criterion == "at_least":
        return min(int(np.searchsorted(cumulative, threshold, side="left") + 1), len(ratios))
    raise ValueError("Criterio PCA desconocido.")


def fit_pca(h: np.ndarray, threshold: float, criterion: str) -> tuple[dict, dict]:
    if h.ndim != 2 or len(h) < 2 or not np.isfinite(h).all():
        raise ValueError("PCA requiere al menos dos observaciones finitas.")
    mean = h.mean(axis=0)
    centered = h - mean
    a = centered / np.sqrt(len(h) - 1)
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    tolerance = np.finfo(float).eps * max(a.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        raise ValueError("PCA no tiene varianza para representar.")
    eigenvalues = singular[:rank]**2
    ratios = eigenvalues / np.sum(singular**2)
    r = select_components(ratios, threshold, criterion)
    # La columna de mayor magnitud se fija positiva para estabilizar el signo.
    vectors = vh[:rank].T.copy()
    for j in range(rank):
        if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
            vectors[:, j] *= -1
    projection = vectors[:, :r]
    arrays = {"hidden_mean": mean, "Vpca": projection,
              "pca_eigenvalues": eigenvalues, "pca_ratios": ratios,
              "pca_singular_values": singular}
    report = {"rank": rank, "n_components": r, "retained_variance": float(ratios[:r].sum()),
              "threshold": threshold, "criterion": criterion}
    return arrays, report


def transform_features(x: np.ndarray, weights: dict) -> np.ndarray:
    """Recibe X ya normalizado con las estadísticas de training."""
    if x.ndim != 2 or x.shape[1] != weights["V"].shape[0]:
        raise ValueError("Dimensión de entrada incompatible con pesos.")
    result = (np.cos(x @ weights["V"]) - weights["hidden_mean"]) @ weights["Vpca"]
    if not np.isfinite(result).all():
        raise ValueError("La transformación SLS/PCA produjo valores no finitos.")
    return result


def transform_raw_features(raw: np.ndarray, weights: dict) -> np.ndarray:
    return transform_features(standardize(raw, weights["input_mean"], weights["input_scale"]), weights)


def train_reduction(cfg: dict, output: Path) -> dict:
    output = Path(output).resolve()
    data, data_meta = load_artifact(output/"dtrain", "dtrain")
    validate_dataset(data, data_meta)
    check_settings(preparation_settings(cfg), data_meta["preparation_settings"], "Preparación")
    x = data["X"]
    sls_arrays, sls_report = fit_sls(x, cfg["sls"])
    h = np.cos(x @ sls_arrays["V"])
    pca_arrays, pca_report = fit_pca(h, cfg["pca"]["variance_threshold"], cfg["pca"]["criterion"])
    settings = {"sls": cfg["sls"], "pca": cfg["pca"]}
    metadata = {"data_run_id": data_meta["data_run_id"], "dtrain_sha256": sha256_file(output/"dtrain"),
                "preparation_settings": data_meta["preparation_settings"],
                "class_names": data_meta["class_names"], "settings": settings,
                "sls": sls_report, "pca": pca_report}
    arrays = {**sls_arrays, **pca_arrays, "input_mean": data["input_mean"], "input_scale": data["input_scale"]}
    save_artifact(output/"pesos", "pesos", metadata, **arrays)
    logs = output.parent/"Trabajo"/"registros"
    logs.mkdir(parents=True, exist_ok=True)
    np.savetxt(logs/"costo_sls.csv", arrays["sls_history"], delimiter=",",
               header="epoch,cost_sum,reconstruction_mse", comments="", fmt="%.17g")
    spectrum = np.column_stack((np.arange(1, len(arrays["pca_ratios"]) + 1),
                                arrays["pca_eigenvalues"], arrays["pca_ratios"],
                                np.cumsum(arrays["pca_ratios"])))
    np.savetxt(logs/"espectro_pca.csv", spectrum, delimiter=",",
               header="component,eigenvalue,ratio,cumulative_ratio", comments="", fmt="%.17g")
    results = output/"Resultados"
    results.mkdir(exist_ok=True)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(arrays["sls_history"][:, 0], arrays["sls_history"][:, 2], color="#176B87")
    ax.scatter([sls_report["selected_epoch"]], [sls_report["reconstruction_mse"]], color="#BA4A00",
               label=f"Modelo guardado: época {sls_report['selected_epoch']}", zorder=3)
    ax.set(xlabel="Época", ylabel="Error cuadrático medio de reconstrucción",
           title="SLS coseno · entrenamiento HP0–HP2")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(results/"grafica_reconstruccion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(spectrum[:, 0], 100 * spectrum[:, 3], "o-", color="#176B87")
    ax.axhline(100 * cfg["pca"]["variance_threshold"], linestyle="--", color="#BA4A00", label="Umbral configurado")
    ax.axvline(pca_report["n_components"], linestyle=":", color="#555555", label="Componentes guardados")
    ax.set(xlabel="Número de componentes", ylabel="Varianza acumulada (%)",
           title="PCA por SVD · criterio de selección " + ("≤ umbral" if cfg["pca"]["criterion"] == "at_most" else "≥ umbral"))
    ax.set_xticks(spectrum[:, 0].astype(int))
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(results/"varianza_pca.png", dpi=160)
    plt.close(fig)
    report = {**metadata, "pesos_sha256": sha256_file(output/"pesos")}
    log_stage(output, "dimr", report)
    print(f"PCA: {h.shape[1]} -> {pca_report['n_components']}; varianza {pca_report['retained_variance']:.2%}.", flush=True)
    return report


def main() -> None:
    args = stage_arguments("Ajustar SLS y PCA en dtrain; guardar pesos.")
    cfg, _ = load_config(args.config)
    train_reduction(cfg, args.output_dir)


if __name__ == "__main__":
    main()
