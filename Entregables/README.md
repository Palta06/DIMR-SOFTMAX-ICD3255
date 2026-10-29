# Evaluación 2: ejecución y resultados

Implementación de **MDE → SLS coseno (RMSprop + RLS) → PCA por SVD → Softmax con RMSprop**. Los cuatro programas exigidos están completos y ejecutados sobre los datos originales. `METODOLOGIA.md` identifica las fórmulas, páginas de los PDF y decisiones del experimento.

Proyecto: `C:\Users\matia\Documents\AID\Eval2-Implementacion`.

## Ejecutar en este computador

El entorno `.venv` ya está instalado. Desde PowerShell:

```powershell
Set-Location 'C:\Users\matia\Documents\AID\Eval2-Implementacion'
& '.\.venv\Scripts\python.exe' '.\Entregables\mdef.py'
& '.\.venv\Scripts\python.exe' '.\Entregables\dimr.py'
& '.\.venv\Scripts\python.exe' '.\Entregables\train.py'
& '.\.venv\Scripts\python.exe' '.\Entregables\test.py'
```

Cada programa se puede invocar con su ruta absoluta desde cualquier directorio. Busca la configuración junto al script, resuelve `data.directory` respecto del archivo de configuración y guarda las salidas junto a los scripts. Al repetir una etapa se reemplazan sus propios resultados; después de cambiar parámetros, ejecutar nuevamente las cuatro etapas.

Para evaluar los modelos ya entrenados basta con el último comando. `test.py` requiere `dtest`, `pesos`, `pesoftmax`, la configuración y los módulos Python; no necesita los CSV originales ni `dtrain`, ni ajusta parámetros.

## Instalar en otro computador

Usar Python **3.14**; el entorno comprobado es CPython 3.14.7, Windows x64. Desde la raíz del proyecto:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r '.\Entregables\requirements.txt'
```

`requirements.txt` fija las versiones directas y transitivas de la ejecución comprobada. La igualdad byte a byte se verificó en este entorno; otra versión de BLAS, sistema o plataforma puede introducir diferencias pequeñas de punto flotante.

## Programas y archivos

| Programa | Lee | Genera |
|---|---|---|
| `mdef.py` | `../data/originales/class1.csv` a `class4.csv` | `dfeatures.csv`, `dclass.csv`, `dtrain`, `dtest` |
| `dimr.py` | `dtrain` | `pesos`, diagnóstico SLS y PCA |
| `train.py` | `dtrain`, `pesos` | `pesoftmax`, `costo`, `Resultados/grafica_costo.png` |
| `test.py` | `dtest`, `pesos`, `pesoftmax` | Matriz de confusión, F1, predicciones y registro de la ejecución |

`comun.py` es un módulo auxiliar compartido: normalización, RMSprop, validaciones y persistencia. Debe acompañar a los cuatro programas. Los imports no ejecutan entrenamiento; cada programa tiene su entrada `main`.

| Artefacto | Formato y contenido |
|---|---|
| `dfeatures.csv` | CSV sin cabecera, 928 × 20: MDE normalizado con estadísticas de training, después del shuffle conjunto. |
| `dclass.csv` | CSV sin cabecera, 928 × 4: etiquetas one-hot en el mismo orden de filas. |
| `dtrain`, `dtest` | Contenedor NumPy NPZ, **sin extensión añadida**: `X`, `T`, `F_raw`, `origins`, `global_row`, `input_mean`, `input_scale`, `metadata`. |
| `pesos` | NPZ sin extensión: `V`, `U`, `hidden_mean`, `Vpca`, espectro PCA, normalización inicial, historial SLS y metadatos. |
| `pesoftmax` | NPZ sin extensión: `W`, `reduced_mean`, `reduced_scale`, acumulador RMSprop, historial y metadatos. |
| `costo` | CSV **sin extensión** con cabecera `epoch,cross_entropy`; incluye inicialización (época 0) y las 500 épocas. |
| `Resultados/metricas.json` | Métricas, matriz, orientación de ejes, nombres de clases y hashes de los modelos. |
| `Resultados/ejecucion.json` | Configuración efectiva y hashes SHA-256 del código y de los 15 archivos numéricos/gráficos de la ejecución. |

Los contenedores usan `allow_pickle=False`. Para inspeccionarlos desde Python:

```python
import numpy as np
with np.load("pesos", allow_pickle=False) as archivo:
    print(archivo.files)
    print(archivo["V"].shape, archivo["Vpca"].shape)
```

Los índices internos de clase son 0–3; las columnas one-hot y las probabilidades siguen `class1`, `class2`, `class3`, `class4`. Las tablas públicas muestran clases 1–4. `origins` contiene clase, columna HP, muestra inicial incluida y muestra final excluida. `global_row` enlaza cada partición con los dos CSV apilados.

## Configuración de esta ejecución

| Parte | Valor |
|---|---|
| Señales | 4 archivos, cada uno de 120000 × 4; columnas HP0–HP3; DE, 12000 Hz. |
| Ventanas | W=2048; paso=2048; 58 ventanas por columna, sin solapamiento; 1216 muestras finales descartadas por columna. |
| MDE | d=3, tau=1, c=3; escalas 1–20; DE normalizada por log2(c^d). |
| Partición | Training HP0–HP2: 696 ventanas, 174 por clase; testing HP3: 232, 58 por clase. Proporción 75/25. |
| Normalización | Z-score por característica; media y desviación muestral (`ddof=1`) solo de training; aplicada antes de SLS y después de PCA. |
| SLS | D=20, J=12; coseno; lambda=0.01; 300 épocas, batch=64, tasa=0.001. |
| PCA | SVD de H centrada / sqrt(N−1); criterio literal de la lámina: mayor r con acumulación ≤ 95 %. |
| Softmax | 4 clases, sesgo incluido; 500 épocas, batch=64, tasa=0.01. |
| RMSprop | beta=0.9, epsilon=1e-8 **fuera de la raíz**, acumulador conservado entre batches y épocas. |
| Semillas | Preparación: 42; SLS: 43; Softmax: 44. |

Se incluye el último batch de 56 muestras. Se guarda la época con menor costo **de entrenamiento**, después de completar todas las épocas. Ningún hiperparámetro fue elegido a partir de HP3.

Los valores de ventana, escalas, partición y optimización son decisiones documentadas del experimento; no se atribuyen al enunciado como valores obligatorios.

## Resultados obtenidos

La reducción final es **20 → 12 → 8**. PCA retiene **92.6582 %** de la varianza de H. El criterio del apunte es ≤ 95 %: este porcentaje no significa que se haya conservado al menos 95 %.

SLS guardó la época 299: costo penalizado de 5782.842649 a 1793.586964 y MSE de reconstrucción final 0.257372. Softmax guardó la época 414, con entropía cruzada media de training **0.493002**.

Prueba HP3: **165/232 aciertos = 71.1207 %**, **F1 macro = 0.703450**.

| Clase | Precisión | Recall | F1 | Ventanas reales |
|---|---:|---:|---:|---:|
| Saludable | 0.512821 | 0.689655 | 0.588235 | 58 |
| Daño pista externa | 0.964286 | 0.931034 | 0.947368 | 58 |
| Daño pista interna | 0.836066 | 0.879310 | 0.857143 | 58 |
| Daño elementos rodantes | 0.540541 | 0.344828 | 0.421053 | 58 |

La matriz usa **filas predichas y columnas reales**, conforme a SOFTMAX p. 16. El principal error es clasificar como saludable 36 de las 58 ventanas con daño en elementos rodantes. Este resultado evalúa transferencia hacia la carga HP3 de estos archivos; no demuestra generalización a otros rodamientos o adquisiciones independientes. Las ventanas de una misma señal pueden estar correlacionadas aun sin solapamiento.

## Verificación y trazabilidad

Pruebas matemáticas y de contratos:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s '.\Trabajo\verificaciones' -v
```

Verificación completa, incluyendo ejecución repetida desde otro directorio:

```powershell
& '.\.venv\Scripts\python.exe' '.\Trabajo\verificaciones\verificar_integracion.py'
```

Se comprobaron **26 pruebas**, incluyendo los dos ejemplos de DE del PDF, diferencias finitas SLS/Softmax, RLS, RMSprop, PCA, orientación de métricas y predicción en un proceso nuevo que solo dispone de los modelos y `dtest`. Los **15 archivos de datos, modelos, curvas, predicciones y métricas** se reprodujeron byte a byte al ejecutar otra vez los cuatro scripts. Los originales siguen coincidiendo con el ZIP.

Los registros están en `../Trabajo/registros`: `procedencia.csv`, costos SLS, espectro PCA, inventario de fuentes, logs de pruebas y `verificacion_final.json`. La repetición se conserva en `../Trabajo/verificaciones/repeticion` y se excluye de la entrega. Estas pruebas verifican la configuración actual; los conteos esperados deben actualizarse deliberadamente si se cambia la ventana o la partición.

## Configuraciones alternativas

Todos los scripts aceptan `--config RUTA_JSON` y `--output-dir CARPETA`. Pasar el mismo directorio de salida a las cuatro etapas. Las rutas de datos se resuelven respecto del JSON, no del directorio actual. Un archivo de configuración copiado a otra carpeta deberá actualizar `data.directory`.

El valor PCA `at_most` conserva el criterio del apunte. `at_least` está disponible para un experimento alternativo explícito; no se utilizó aquí. Si el criterio `at_most` no permite conservar ningún componente, el programa detiene la ejecución y pide revisar el umbral, sin cambiar silenciosamente de criterio.

Para comparar hiperparámetros en el futuro, reservar validación dentro de HP0–HP2 y ajustar todas las transformaciones con el entrenamiento interno. HP3 ya fue observado; no utilizar su resultado para seleccionar una variante y presentarla como evaluación independiente.

## Entrega

`Entregables.zip`, ubicado en la raíz del proyecto, contiene esta carpeta, incluidos `comun.py`, configuración, documentación, modelos y resultados. Excluye `.venv`, cachés, originales y experimentos temporales. Para repetir toda la extracción se necesitan además `data/originales`; para ejecutar solo `test.py`, los artefactos incluidos bastan.

`Plan-Evaluacion2.md` y `Plan-Evaluacion2.pdf` en la raíz se conservan como el plan previo. Sus consultas pendientes se resuelven en `METODOLOGIA.md`; el estado final de los doce pasos está en `../IMPLEMENTACION.md`.
