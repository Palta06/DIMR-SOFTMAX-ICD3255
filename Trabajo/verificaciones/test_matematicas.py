"""Ejecutar: python -m unittest discover -s Trabajo/verificaciones -v.

Pruebas independientes: ejemplos oficiales, derivadas por diferencias finitas,
identidades algebraicas y contratos de la ejecución real.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Entregables"
sys.path.insert(0, str(BASE))

from comun import (array_digest, fit_standardizer, load_artifact, load_config,
                   rmsprop_step, save_artifact, sha256_file, standardize)
from mdef import (coarse_grain, dispersion_entropy, dispersion_symbols,
                  multiscale_dispersion_entropy)
from dimr import (fit_pca, select_components, sls_cost, sls_gradient,
                  solve_rls, transform_features, transform_raw_features)
from train import design_matrix, loss_and_gradient, softmax_probabilities
from test import classification_metrics, confusion_matrix, evaluate, predict


def finite_gradient(function, weights, step=1e-6):
    result = np.empty_like(weights)
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


class EntropyTests(unittest.TestCase):
    def test_example_dispersion_page_8(self):
        x = np.array([3, 11, 4, 4, 7, 12, 7, 8, 11, 3, 9, 1])
        np.testing.assert_array_equal(dispersion_symbols(x, 3), [1, 3, 1, 1, 2, 3, 2, 2, 3, 1, 3, 1])
        self.assertAlmostEqual(dispersion_entropy(x, 2, 1, 3, False), 2.6635327548042547, places=12)
        self.assertAlmostEqual(dispersion_entropy(x, 2, 1, 3), .8402510323090768, places=12)

    def test_example_dispersion_page_11(self):
        x = np.array([20, 5, 21, 1, 8, 1, 3, 25, 21, 10, 29, 1, 20, 5, 21])
        self.assertAlmostEqual(dispersion_entropy(x, 3, 1, 3, False), 2.5654483718208256, places=12)
        self.assertAlmostEqual(dispersion_entropy(x, 3, 1, 3), .5395392363444035, places=12)

    def test_endpoints_constant_and_uniform(self):
        np.testing.assert_array_equal(dispersion_symbols(np.array([0., .5, 1.]), 3), [1, 2, 3])
        self.assertEqual(dispersion_entropy(np.ones(20), 3, 1, 3), 0)
        self.assertAlmostEqual(dispersion_entropy(np.array([0, .5, 1]), 1, 1, 3), 1)

    def test_coarse_grain_and_independent_scales(self):
        x = np.arange(1, 21, dtype=float)
        np.testing.assert_array_equal(coarse_grain(x, 3), [2, 5, 8, 11, 14, 17])
        expected = [dispersion_entropy(x, 2, 1, 3),
                    dispersion_entropy(np.arange(1.5, 20, 2), 2, 1, 3),
                    dispersion_entropy(np.array([2, 5, 8, 11, 14, 17]), 2, 1, 3)]
        np.testing.assert_allclose(multiscale_dispersion_entropy(x, 2, 1, 3, 3), expected)

    def test_delay_changes_pattern_count_and_bounds(self):
        # d=2, delay=2: patrones (1,1),(3,3),(1,1),(3,3), equiprobables.
        x = np.array([0, 1, 0, 1, 0, 1])
        self.assertAlmostEqual(dispersion_entropy(x, 2, 2, 3, False), 1)
        values = multiscale_dispersion_entropy(np.random.default_rng(1).normal(size=2048), 3, 1, 3, 20)
        self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_invalid_series_and_insufficient_patterns(self):
        for x in (np.array([1., np.nan, 2.]), np.array([1., np.inf, 2.]), np.array([1., 2.])):
            with self.assertRaises(ValueError):
                dispersion_entropy(x, 3, 1, 3)


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(8, 4))
        self.v = rng.normal(size=(4, 3))
        self.u = rng.normal(size=(3, 4))

    def test_sls_gradient_with_fixed_u(self):
        numeric = finite_gradient(lambda v: sls_cost(self.x, v, self.u, .02), self.v)
        np.testing.assert_allclose(sls_gradient(self.x, self.v, self.u), numeric, rtol=2e-6, atol=1e-7)

    def test_sls_profile_gradient_including_rls(self):
        def profile(v):
            u = solve_rls(np.cos(self.x @ v), self.x, .02)
            return sls_cost(self.x, v, u, .02)
        u = solve_rls(np.cos(self.x @ self.v), self.x, .02)
        np.testing.assert_allclose(sls_gradient(self.x, self.v, u), finite_gradient(profile, self.v), rtol=2e-5, atol=1e-7)

    def test_rls_normal_equations_and_minimum(self):
        h = np.cos(self.x @ self.v)
        u = solve_rls(h, self.x, .02)
        np.testing.assert_allclose((h.T @ h + .02*np.eye(3)) @ u, h.T @ self.x, atol=1e-12)
        optimum = sls_cost(self.x, self.v, u, .02)
        for delta in (.01, -.01, 1):
            self.assertLess(optimum, sls_cost(self.x, self.v, u + delta * self.u, .02))

    def test_rmsprop_hand_calculation_epsilon_outside_sqrt(self):
        theta = np.array([1., -2.])
        gradient = np.array([2., -3.])
        result, acc = rmsprop_step(theta, gradient, np.zeros(2), .1, .9, .01)
        np.testing.assert_allclose(acc, [.4, .9], atol=1e-15)
        np.testing.assert_allclose(result, [1 - .2/(np.sqrt(.4)+.01), -2 + .3/(np.sqrt(.9)+.01)], atol=1e-15)
        _, acc2 = rmsprop_step(result, gradient, acc, .1, .9, .01)
        np.testing.assert_allclose(acc2, [.76, 1.71], atol=1e-15)

    def test_softmax_gradient(self):
        q = np.column_stack((np.ones(8), self.x))
        w = np.random.default_rng(2).normal(size=(5, 4))
        t = np.eye(4)[np.arange(8) % 4]
        _, analytic = loss_and_gradient(q, t, w)
        numeric = finite_gradient(lambda a: loss_and_gradient(q, t, a)[0], w)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=1e-8)

    def test_softmax_uniform_shift_and_extreme_scores(self):
        np.testing.assert_allclose(softmax_probabilities(np.zeros((3, 4))), .25)
        scores = np.array([[1000., 999., 998., 997.], [-1000., 1000., 0., 30.]])
        p = softmax_probabilities(scores)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-15)
        np.testing.assert_allclose(p, softmax_probabilities(scores + 321), atol=1e-15)
        q, t, w = np.eye(2), np.eye(4)[[0, 1]], scores
        self.assertTrue(np.isfinite(loss_and_gradient(q, t, w)[0]))
        self.assertAlmostEqual(loss_and_gradient(q, t, np.zeros((2, 4)))[0], np.log(4))


class PCATests(unittest.TestCase):
    def test_course_threshold_example(self):
        accumulated = np.array([42.3620, 63.2424, 78.8556, 93.0828, 99.0177, 100.0000]) / 100
        ratios = np.diff(np.r_[0, accumulated])
        self.assertEqual(select_components(ratios, .95, "at_most"), 4)
        self.assertEqual(select_components(ratios, .95, "at_least"), 5)
        with self.assertRaises(ValueError):
            select_components(np.array([.99, .01]), .95, "at_most")

    def test_svd_against_covariance_eigendecomposition(self):
        h = np.random.default_rng(12).normal(size=(40, 5)) * np.array([5, 3, 2, 1, .1]) + 7
        weights, report = fit_pca(h, .95, "at_most")
        covariance = np.cov(h, rowvar=False, ddof=1)
        values, vectors = np.linalg.eigh(covariance)
        np.testing.assert_allclose(weights["pca_eigenvalues"], values[::-1], rtol=1e-12)
        r, p = report["n_components"], weights["Vpca"]
        np.testing.assert_allclose(p.T @ p, np.eye(r), atol=1e-12)
        expected = vectors[:, -r:]
        np.testing.assert_allclose(p @ p.T, expected @ expected.T, atol=1e-12)
        scores = (h - weights["hidden_mean"]) @ p
        np.testing.assert_allclose(scores.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.cov(scores, rowvar=False), np.diag(values[::-1][:r]), atol=1e-12)

    def test_projection_error_decreases_and_zero_variance_fails(self):
        h = np.random.default_rng(4).normal(size=(25, 4))
        weights, _ = fit_pca(h, 1, "at_most")
        centered, p = h - weights["hidden_mean"], weights["Vpca"]
        errors = [np.linalg.norm(centered - centered @ p[:, :r] @ p[:, :r].T)**2 for r in range(1, 5)]
        self.assertTrue(np.all(np.diff(errors) <= 1e-12))
        self.assertLess(errors[-1], 1e-20)
        with self.assertRaises(ValueError):
            fit_pca(np.ones((10, 3)), .95, "at_most")


class MetricsAndStorageTests(unittest.TestCase):
    def test_ejercicios_classification_example(self):
        predicted = np.array([2, 2, 2, 3, 3, 3, 4, 4, 3, 2]) - 1
        real = np.array([4, 2, 2, 3, 3, 3, 4, 4, 3, 2]) - 1
        matrix = confusion_matrix(predicted, real)
        np.testing.assert_array_equal(matrix, [[0, 0, 0, 0], [0, 3, 0, 1], [0, 0, 4, 0], [0, 0, 0, 2]])
        metrics = classification_metrics(matrix)
        np.testing.assert_allclose(metrics["precision"], [0, .75, 1, 1])
        np.testing.assert_allclose(metrics["recall"], [0, 1, 1, 2/3])
        np.testing.assert_allclose(metrics["f1"], [0, 6/7, 1, .8])
        self.assertEqual(metrics["accuracy"], .9)

    def test_perfect_classification(self):
        metrics = classification_metrics(np.diag([2, 3, 4, 5]))
        np.testing.assert_array_equal(metrics["f1"], np.ones(4))
        self.assertEqual(metrics["accuracy"], 1)

    def test_normalizer_reuses_training_and_constant_column(self):
        train = np.array([[1., 5.], [3., 5.], [5., 5.]])
        mean, scale = fit_standardizer(train)
        np.testing.assert_array_equal(mean, [3, 5])
        np.testing.assert_array_equal(scale, [2, 1])
        np.testing.assert_array_equal(standardize(np.array([[103., 5.]]), mean, scale), [[50, 0]])

    def test_extensionless_serialization_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pesos"
            save_artifact(path, "pesos", {"source": "test"}, A=np.eye(3))
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["pesos"])
            arrays, meta = load_artifact(path, "pesos")
            np.testing.assert_array_equal(arrays["A"], np.eye(3))
            with path.open("wb") as stream:
                np.savez(stream, metadata=np.array(json.dumps(meta)), A=np.zeros((3, 3)))
            with self.assertRaisesRegex(ValueError, "integridad"):
                load_artifact(path, "pesos")


class RealPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = load_config()
        cls.train, cls.tm = load_artifact(BASE/"dtrain", "dtrain")
        cls.test, cls.em = load_artifact(BASE/"dtest", "dtest")
        cls.reduction, cls.rm = load_artifact(BASE/"pesos", "pesos")
        cls.classifier, cls.cm = load_artifact(BASE/"pesoftmax", "pesoftmax")

    def test_split_no_shared_sources_and_alignment(self):
        a, b = self.train, self.test
        train_groups = set(map(tuple, a["origins"][:, :2]))
        test_groups = set(map(tuple, b["origins"][:, :2]))
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(set(a["origins"][:, 1]), {0, 1, 2})
        self.assertEqual(set(b["origins"][:, 1]), {3})
        all_x, all_t = np.loadtxt(BASE/"dfeatures.csv", delimiter=","), np.loadtxt(BASE/"dclass.csv", delimiter=",")
        self.assertEqual((len(a["X"]), len(b["X"]), len(all_x)), (696, 232, 928))
        np.testing.assert_array_equal(np.sort(np.r_[a["global_row"], b["global_row"]]), np.arange(928))
        for data in (a, b):
            np.testing.assert_array_equal(all_x[data["global_row"]], data["X"])
            np.testing.assert_array_equal(all_t[data["global_row"]], data["T"])
            np.testing.assert_array_equal(data["T"].argmax(axis=1), data["origins"][:, 0])
            self.assertTrue(np.all(data["origins"][:, 3] - data["origins"][:, 2] == 2048))

    def test_train_only_normalization_and_raw_inference(self):
        mean, scale = fit_standardizer(self.train["F_raw"])
        np.testing.assert_array_equal(mean, self.train["input_mean"])
        np.testing.assert_array_equal(scale, self.test["input_scale"])
        np.testing.assert_allclose(self.train["X"].mean(axis=0), 0, atol=5e-14)
        np.testing.assert_allclose(self.train["X"].std(axis=0, ddof=1), 1, atol=1e-13)
        np.testing.assert_allclose(transform_raw_features(self.test["F_raw"], self.reduction),
                                   transform_features(self.test["X"], self.reduction), atol=1e-14)
        reduced = transform_features(self.train["X"], self.reduction)
        mean, scale = fit_standardizer(reduced)
        np.testing.assert_array_equal(mean, self.classifier["reduced_mean"])
        np.testing.assert_array_equal(scale, self.classifier["reduced_scale"])

    def test_saved_sls_and_pca(self):
        x, w = self.train["X"], self.reduction
        h = np.cos(x @ w["V"])
        left = (h.T @ h + self.cfg["sls"]["lambda"] * np.eye(h.shape[1])) @ w["U"]
        np.testing.assert_allclose(left, h.T @ x, atol=2e-10, rtol=2e-10)
        self.assertLess(self.rm["sls"]["selected_cost"], self.rm["sls"]["initial_cost"])
        self.assertAlmostEqual(sls_cost(x, w["V"], w["U"], self.cfg["sls"]["lambda"]), self.rm["sls"]["selected_cost"], places=9)
        p = w["Vpca"]
        np.testing.assert_allclose(p.T @ p, np.eye(p.shape[1]), atol=1e-12)
        reduced = transform_features(x, w)
        np.testing.assert_allclose(np.cov(reduced, rowvar=False), np.diag(w["pca_eigenvalues"][:p.shape[1]]), atol=1e-12)

    def test_cost_matches_saved_weights_and_history(self):
        z = transform_features(self.train["X"], self.reduction)
        z = standardize(z, self.classifier["reduced_mean"], self.classifier["reduced_scale"])
        cost, _ = loss_and_gradient(design_matrix(z, True), self.train["T"], self.classifier["W"])
        self.assertAlmostEqual(cost, self.cm["training"]["selected_cost"], places=14)
        self.assertLess(cost, self.cm["training"]["initial_cost"])
        np.testing.assert_array_equal(np.loadtxt(BASE/"costo", delimiter=",", skiprows=1), self.classifier["cost_history"])

    def test_metrics_match_predictions(self):
        predicted, p = predict(self.test["X"], self.reduction, self.classifier, True)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        matrix = confusion_matrix(predicted, self.test["T"].argmax(axis=1))
        summary = json.loads((BASE/"Resultados/metricas.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(matrix, summary["confusion_matrix"])
        self.assertEqual(int(matrix.sum()), 232)
        self.assertAlmostEqual(classification_metrics(matrix)["macro_f1"], summary["macro_f1"])

    def test_inference_without_training_data_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"Entregables"
            output.mkdir()
            for name in ("dtest", "pesos", "pesoftmax"):
                (output/name).write_bytes((BASE/name).read_bytes())
            before = {name: sha256_file(output/name) for name in ("dtest", "pesos", "pesoftmax")}
            result = subprocess.run([sys.executable, str(BASE/"test.py"), "--output-dir", str(output)],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output/"Resultados/metricas.json").read_text(encoding="utf-8"))
            original = json.loads((BASE/"Resultados/metricas.json").read_text(encoding="utf-8"))
            self.assertEqual(summary, original)
            self.assertEqual(before, {name: sha256_file(output/name) for name in before})

    def test_mismatched_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"Entregables"
            output.mkdir()
            for name in ("dtest", "pesoftmax"):
                (output/name).write_bytes((BASE/name).read_bytes())
            modified = {key: value.copy() for key, value in self.reduction.items()}
            modified["V"][0, 0] += .001
            save_artifact(output/"pesos", "pesos", self.rm, **modified)
            with self.assertRaisesRegex(ValueError, "no corresponde"):
                evaluate(self.cfg, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
