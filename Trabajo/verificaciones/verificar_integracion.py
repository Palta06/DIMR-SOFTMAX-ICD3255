"""Auditar los originales, ejecutar pruebas y repetir toda la cadena en otra carpeta.

No cambia parámetros ni selecciona modelos usando testing. La segunda ejecución
comprueba determinismo de un experimento previamente fijado.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
import zipfile
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"Entregables"
sys.path.insert(0, str(BASE))
from comun import save_json, sha256_file


def main() -> None:
    logs = ROOT/"Trabajo"/"registros"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE/"Resultados"/"ejecucion.json").read_text(encoding="utf-8"))
    for name, expected in manifest["code_sha256"].items():
        if sha256_file(BASE/name) != expected:
            raise RuntimeError(f"Cambió {name} desde la ejecución registrada: ejecute los cuatro scripts.")
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(BASE/name) != expected:
            raise RuntimeError(f"El archivo {name} no corresponde a la ejecución registrada.")
    originals = {}
    with zipfile.ZipFile(ROOT/"data"/"DATA.zip") as archive:
        for name in ["class1.csv", "class2.csv", "class3.csv", "class4.csv", "FormatData.pdf"]:
            actual = sha256_file(ROOT/"data"/"originales"/name)
            if actual != hashlib.sha256(archive.read(name)).hexdigest():
                raise RuntimeError(f"El original {name} difiere del ZIP.")
            originals[name] = actual
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    (logs/"pruebas_matematicas.log").write_text(capture.getvalue(), encoding="utf-8")
    if not result.wasSuccessful():
        print(capture.getvalue())
        raise RuntimeError("Falló una comprobación: consulte pruebas_matematicas.log.")
    repeated = Path(__file__).parent/"repeticion"/"Entregables"
    repeated.mkdir(parents=True, exist_ok=True)
    for stage in ("mdef", "dimr", "train", "test"):
        process = subprocess.run([sys.executable, str(BASE/f"{stage}.py"), "--output-dir", str(repeated)],
                                 cwd=Path(__file__).parent, capture_output=True, encoding="utf-8", errors="replace")
        (logs/f"repeticion_{stage}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"Falló la repetición de {stage}; consulte su registro.")
        print(f"Repetición independiente: {stage}.py OK", flush=True)
    compared = {}
    # Datos, modelos, curvas, predicciones, tablas e imágenes deben ser idénticos.
    for name, expected in manifest["files_sha256"].items():
        actual = sha256_file(repeated/name)
        if actual != expected:
            raise RuntimeError(f"La repetición produjo diferencias en {name}.")
        compared[name] = actual
    for name, expected in originals.items():
        if sha256_file(ROOT/"data"/"originales"/name) != expected:
            raise RuntimeError(f"Se modificó el original {name} durante la ejecución.")
    report = {"status": "passed", "tests_run": result.testsRun, "tests_failed": len(result.failures),
              "tests_errors": len(result.errors), "tests_skipped": len(result.skipped),
              "repeated_stages": ["mdef.py", "dimr.py", "train.py", "test.py"],
              "data_run_id": manifest["data_run_id"], "originals_unchanged_sha256": originals,
              "reproduced_files_sha256": compared,
              "fresh_process_inference_without_training_data": True,
              "independent_working_directory": True}
    save_json(logs/"verificacion_final.json", report)
    print(f"VERIFICADO: {result.testsRun} pruebas; {len(compared)} archivos reproducidos byte a byte; originales intactos.")


if __name__ == "__main__":
    main()
