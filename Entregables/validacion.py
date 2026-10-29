"""Selección de J (neuronas SLS) y del criterio PCA mediante validación cruzada por carga.

Script de apoyo (no es uno de los cuatro exigidos). Usa SOLO training (HP0-HP2):
en cada pliegue se entrena con dos cargas y se valida con la tercera; la
normalización, SLS, PCA y Softmax se reajustan con las dos cargas internas.
HP3 (testing) NO se usa. Escribe Resultados/validacion_cv.csv y el registro
Trabajo/registros/validacion_cv.json. Para aplicar la mejor configuración, copie
sus valores a configuracion.json (sls.latent_dim y pca) y ejecute mdef, dimr, train y test.

Requiere haber ejecutado antes mdef.py.
"""
from __future__ import annotations

import contextlib
import io
import itertools
from pathlib import Path

import numpy as np

from comun import (fit_standardizer, load_artifact, load_config, log_stage,
                   save_json, stage_arguments, standardize)
from dimr import fit_pca, fit_sls, transform_features
from train import design_matrix, fit_softmax

LATENT_GRID = [8, 12, 14, 16, 18, 19]
PCA_GRID = [("at_most", 0.95), ("at_least", 0.95), ("at_least", 0.99)]


def fold_accuracy(f_in, t_in, f_val, t_val, cfg, latent, criterion, threshold):
    ddof = cfg["normalization"]["ddof"]
    mean, scale = fit_standardizer(f_in, ddof)
    x_in, x_val = standardize(f_in, mean, scale), standardize(f_val, mean, scale)
    sls = {**cfg["sls"], "latent_dim": latent}
    with contextlib.redirect_stdout(io.StringIO()):
        arrays, _ = fit_sls(x_in, sls)
    pca, report = fit_pca(np.cos(x_in @ arrays["V"]), threshold, criterion)
    weights = {**arrays, **pca}
    z_in, z_val = transform_features(x_in, weights), transform_features(x_val, weights)
    m2, s2 = fit_standardizer(z_in, ddof)
    q_in = design_matrix(standardize(z_in, m2, s2), cfg["softmax"]["intercept"])
    q_val = design_matrix(standardize(z_val, m2, s2), cfg["softmax"]["intercept"])
    with contextlib.redirect_stdout(io.StringIO()):
        soft, _ = fit_softmax(q_in, t_in, cfg["softmax"])
    predicted = (q_val @ soft["W"]).argmax(axis=1)
    return float(np.mean(predicted == t_val.argmax(axis=1))), report["n_components"]


def main() -> None:
    args = stage_arguments("Validación cruzada por carga (solo training) para elegir J y criterio PCA.")
    cfg, _ = load_config(args.config)
    output = Path(args.output_dir).resolve()
    data, _ = load_artifact(output / "dtrain", "dtrain")
    f_raw, targets, load = data["F_raw"], data["T"], data["origins"][:, 1]
    folds = sorted(set(load.tolist()))
    rows = []
    for latent, (criterion, threshold) in itertools.product(LATENT_GRID, PCA_GRID):
        scores, comps = [], []
        for fold in folds:
            inner, held = load != fold, load == fold
            acc, r = fold_accuracy(f_raw[inner], targets[inner], f_raw[held], targets[held],
                                   cfg, latent, criterion, threshold)
            scores.append(acc)
            comps.append(r)
        rows.append({"latent_dim": latent, "criterion": criterion, "threshold": threshold,
                     "mean_accuracy": float(np.mean(scores)), "std_accuracy": float(np.std(scores)),
                     "fold_accuracy": scores, "components_per_fold": comps})
        print(f"J={latent:2d} {criterion:8s} {threshold:.2f}: CV={np.mean(scores):.4f} "
              f"(r por pliegue {comps})", flush=True)
    # Mejor exactitud media; empate -> menor J, luego criterio literal at_most.
    best = max(rows, key=lambda r: (round(r["mean_accuracy"], 6), -r["latent_dim"], r["criterion"] == "at_most"))
    results = output / "Resultados"
    results.mkdir(exist_ok=True)
    with (results / "validacion_cv.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("latent_dim,criterion,threshold,mean_accuracy,std_accuracy,fold_accuracies,components_per_fold\n")
        for r in rows:
            stream.write(f"{r['latent_dim']},{r['criterion']},{r['threshold']},{r['mean_accuracy']:.6f},"
                         f"{r['std_accuracy']:.6f},{' '.join(f'{a:.4f}' for a in r['fold_accuracy'])},"
                         f"{' '.join(map(str, r['components_per_fold']))}\n")
    summary = {"folds": [f"HP{c}" for c in folds], "grid": rows, "selected": best,
               "uses_testing": False}
    log_stage(output, "validacion_cv", summary)
    print(f"\nSeleccionado: J={best['latent_dim']}, PCA {best['criterion']} {best['threshold']} "
          f"-> CV={best['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()