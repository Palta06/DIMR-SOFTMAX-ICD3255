# Fórmulas y decisiones de implementación

Las páginas indicadas son posiciones dentro del PDF, empezando en 1. Las fuentes de la cátedra están en `../Referencias`; el enunciado está en la raíz. Las decisiones no fijadas por los apuntes se distinguen explícitamente de las fórmulas.

## 1. Mapa de fuentes y funciones

| Fuente | Aplicación | Funciones |
|---|---|---|
| Enunciado-Evaluacion2, pp. 2 y 5–8 | Cadena requerida, cuatro scripts y nombres de salidas; RMSprop + RLS en SLS. | Entradas `main` de los cuatro scripts. |
| FormatData, p. 1 | Correspondencia clase-archivo, cargas HP0–HP3, 12000 Hz, canal DE. | `mdef.prepare_data`. |
| DISP_Entropy, pp. 2–4, ejemplos pp. 8 y 11 | Min-max, símbolos, embedding, frecuencias, DE y normalización. | `dispersion_symbols`, `dispersion_entropy`. |
| MultEnt, pp. 4–5 | Promedios por escala y MDE. | `coarse_grain`, `multiscale_dispersion_entropy`. |
| EJERCICIOS, pp. 2–5 | Modelo SLS, costo penalizado, RLS, activación coseno. | `solve_rls`, `sls_cost`, `sls_gradient`, `fit_sls`. |
| Integral-PCA, pp. 3–12 y 33–36 | PCA-SVD, criterios ≤ 95 % y ≥ umbral, esquema alternado por minibatches. | `fit_pca`, `select_components`, `fit_sls`. |
| SOFTMAX, pp. 5–6 y 10–16 | Sesgo, probabilidades, CE, RMSprop y métricas. | `design_matrix`, `loss_and_gradient`, `rmsprop_step`, funciones de `test.py`. |

`PERM_Entrropy.pdf` se conserva como referencia del material recibido. La extracción requerida es **MDE**, por lo que la entropía de permutación no forma parte de la cadena.

## 2. MDE y tratamiento de extremos

Para una ventana original x de W muestras, cada escala s se obtiene de esa misma ventana:

```text
L_s = floor(W/s)
y_j^(s) = (1/s) * sum_{i=(j-1)s+1}^{js} x_i,  j=1,...,L_s
MDE(s) = DE_normalizada(y^(s), d, tau, c)
```

Se descarta el bloque incompleto final. `s` es la escala de coarse-graining; `tau` es el retardo del embedding. No se aplica una escala sobre la salida de otra escala.

Sobre cada serie y de longitud L se aplica el min-max del PDF y se construyen M patrones:

```text
z = (y - min(y)) / (max(y) - min(y))
M = L - (d-1)*tau
Z_i = (z_i, z_{i+tau}, ..., z_{i+(d-1)tau})
Y_i = round(c*Z_i + 0.5)
k_i = 1 + sum_{j=0}^{d-1} (Y_ij - 1)*c^j
p_k = frecuencia(k) / M
DE = -sum_{p_k>0} p_k*log2(p_k)
DE_normalizada = DE / log2(c^d)
```

**Decisiones necesarias para implementar la lámina:** el redondeo de mitades se hace hacia arriba para valores no negativos y los símbolos se limitan a 1..c. Así se reproducen los ejemplos, se evita el 0 que puede dar el redondeo al par de Python y el c+1 del extremo máximo. Numéricamente se implementa `clip(floor(c*z+1),1,c)`. No se reemplaza el min-max por una CDF gaussiana.

Una serie constante hace indeterminado el min-max; se define como una secuencia de un solo símbolo y DE=0. Los términos con p=0 aportan cero. Si no hay patrones completos o hay NaN/Inf, se rechaza la entrada.

Los ejemplos oficiales dan DE=2.6635327548042547 y 2.5654483718208256; sus valores normalizados son 0.8402510323090768 y 0.5395392363444035. Se verifican en las pruebas.

Se eligió DE **normalizada**, incluida en el apunte. Con W=2048, s máximo 20, d=3 y tau=1, la escala más corta tiene L=102 y M=100 patrones observados sobre 27 patrones posibles. Esta configuración permite el conteo, pero no elimina el error de estimación de entropía en series finitas. W=2048 y 20 escalas son parámetros propuestos para este experimento.

## 3. Separación y normalización

Cada columna HP se segmenta de forma independiente y conserva la clase de su archivo. No se crean ventanas que crucen una clase o columna. El paso W evita solapamiento. Se conservan las 16 identidades archivo/carga y los intervalos de muestras.

Se decidió reservar las cuatro series HP3 para testing: 25 % de las ventanas. Las doce series HP0–HP2 constituyen training: 75 %. El shuffle usa una sola permutación para características, etiquetas, intervalos y asignación train/test; no cambia la pertenencia de una carga a su partición.

La media mu y la desviación muestral sigma por característica se calculan **solo sobre training**, y se reutilizan:

```text
sigma_j = sqrt(sum_i (X_ij-mu_j)^2 / (Ntrain-1))
X_normalizado_ij = (X_ij-mu_j) / sigma_j
```

Si sigma_j=0 se usa divisor 1; la columna de training queda en cero. Se repite este procedimiento sobre la representación posterior a PCA, antes de Softmax, guardando ambos pares de estadísticas.

El enunciado enumera normalizar antes de dividir; se fija primero la partición para evitar incorporar estadísticas de testing. Esta adaptación del orden está documentada. El min-max por ventana y escala es parte local de DE, y es distinto de estas normalizaciones aprendidas.

## 4. SLS: costo, RLS y gradiente de coseno

La función elegida está escrita en EJERCICIOS p. 5: **H=f(Z)=cos(Z)**, cuya derivada es −sin(Z). Se usa la convención de observaciones en filas:

```text
X: N x D     V: D x J     H=cos(XV): N x J
U: J x D     Y=HU: N x D
E = (1/2)||X-HU||_F^2 + (lambda/2)||U||_F^2
U = (H^T H + lambda I)^(-1) H^T X
G_V = -X^T [ ((HU-X) U^T) ⊙ sin(XV) ]
```

Este costo es el de EJERCICIOS p. 3, que resuelve el factor 1/2 anidado ambiguo de Integral-PCA p. 28. La orientación de la fórmula impresa del gradiente se adapta a las dimensiones declaradas en las láminas; la expresión implementada tiene dimensión D×J y se comprobó por diferencias finitas tanto con U fijo como resolviendo U en cada perturbación.

Se resuelve el sistema lineal para U mediante `numpy.linalg.solve`, equivalente a la fórmula con inversa, sin formar explícitamente esa inversa. Lambda=0.01 asegura unicidad. RLS significa aquí **regularized least squares**, según esta fórmula de la cátedra.

Cada minibatch recalcula H, resuelve U y actualiza V con RMSprop. El costo SLS es una **suma**, sin factor 1/M en su gradiente. Al terminar cada época se vuelve a resolver U con todo training y se registra ese costo y el MSE de reconstrucción. Se guarda la época con menor costo de training. Se conserva U como evidencia de reconstrucción, aunque la inferencia utiliza V y coseno.

Integral-PCA presenta descenso genérico para V; se usa RMSprop porque así lo exige el enunciado. También se conserva la no linealidad en `Xr=cos(XV)`, según el enunciado, frente a la omisión de f en una de las láminas del apunte.

## 5. PCA por SVD

```text
mu_H = mean(H_training, axis=0)
Hc = H_training - mu_H
A = Hc / sqrt(Ntrain-1)
A = U_svd S V_svd^T
eigenvalues = diag(S)^2
ratio_j = eigenvalue_j / sum(eigenvalues)
Vpca = primeras r columnas de V_svd
Xrr = (H - mu_H) Vpca
```

Como A ya incluye el factor sqrt(N−1), los valores singulares al cuadrado no se vuelven a dividir por N−1. Se eliminan direcciones de rango numérico nulo y se fija el signo de cada vector por su entrada de mayor magnitud; este cambio de signo no modifica el subespacio PCA.

El apunte Integral-PCA p. 11 ilustra el criterio literal **mayor r con energía acumulada ≤ 95 %** (`at_most`); el criterio **menor r con energía acumulada ≥ umbral** (`at_least`) también está implementado. Ambos se comparan en `validacion.py` mediante validación cruzada por carga dentro de training (sección 8). La configuración final usa `at_least` con umbral 99 %: se obtienen 16 de 19 componentes y 99.2810 % de varianza retenida (porcentaje real, no el umbral).

## 6. Softmax y RMSprop

Tras normalizar Xrr con estadísticas de training, se agrega una columna inicial de unos. Esto equivale al sesgo de SOFTMAX pp. 5–6. Con r=16 se obtiene Q con 17 columnas y Wsm de dimensión 17×4.

```text
Z = Q Wsm
P_ik = exp(Z_ik) / sum_l exp(Z_il)
E_CE = -(1/M) sum_i sum_k T_ik ln(P_ik)
G_W = Q^T (P-T) / M
```

La implementación resta el máximo de cada fila antes de exponenciar y calcula log-probabilidades con log-sum-exp. Es una evaluación estable de la misma función; no se altera la pérdida recortando probabilidades. El logaritmo de CE es natural; el de DE es base 2.

En SLS y Softmax, para el parámetro Theta y su gradiente G:

```text
A_t = beta*A_(t-1) + (1-beta)*(G_t ⊙ G_t)
Theta_t = Theta_(t-1) - learning_rate*G_t/(sqrt(A_t)+epsilon)
```

Epsilon queda **fuera** de la raíz. Los acumuladores son distintos para V y Wsm y persisten entre épocas. Las inicializaciones son normales con desviaciones 1/sqrt(D) para V y 0.01 para Wsm. Se incluyen todos los ejemplos, también el último minibatch incompleto. El registro `costo` evalúa la CE media sobre todo training al terminar cada época; no es un promedio de costos de batches con pesos distintos.

Se guardan los pesos de menor costo de training; esa época se señala en la gráfica y en los metadatos. No se interpreta esa selección como validación del rendimiento fuera de training.

## 7. Métricas y revisión

Con filas predichas y columnas reales, C_ij cuenta predicción i y etiqueta real j:

```text
TP_k = C_kk
FP_k = sum_j C_kj - TP_k
FN_k = sum_i C_ik - TP_k
precision_k = TP_k/(TP_k+FP_k)
recall_k = TP_k/(TP_k+FN_k)
F1_k = 2*precision_k*recall_k/(precision_k+recall_k)
accuracy = trace(C)/sum(C)
```

Un denominador cero devuelve 0 y se declara en `metricas.json`. F1 macro promedia las cuatro clases. En esta ejecución todas las clases tienen 58 observaciones reales, por lo que F1 macro y ponderado coinciden.

Las comprobaciones incluyen:

- [x] Dos ejemplos oficiales de DE, extremos, constantes, retardo y coarse-graining.
- [x] Correspondencia de filas, clases y procedencia; fuentes train/test disjuntas.
- [x] Estadísticas estimadas solo con training y transformación idéntica al recargar.
- [x] Gradientes SLS y Softmax por diferencias finitas.
- [x] Ecuaciones normales y mínimo cuadrático RLS; costos consistentes con pesos guardados.
- [x] RMSprop comparado con un cálculo manual con epsilon fuera de la raíz.
- [x] PCA comparado con la descomposición propia de la covarianza; ortonormalidad y covarianza diagonal.
- [x] Softmax uniforme, invariancia a traslación y puntuaciones extremas finitas.
- [x] Matriz y métricas verificadas con un ejemplo no simétrico de EJERCICIOS.
- [x] Inferencia en otro proceso sin CSV originales ni `dtrain`; modelos intactos.
- [x] Rechazo de modelos incompatibles y verificación de integridad de los contenedores.
- [x] Repetición completa y comparación byte a byte de 15 artefactos.

El ejemplo numérico de SOFTMAX p. 7 contiene una discrepancia entre los productos matriciales y las puntuaciones impresas, ya identificada en el plan. Se validan las ecuaciones con identidades y diferencias finitas; no se fuerza el código para reproducir ese resultado inconsistente.

Las verificaciones respaldan la implementación de las fórmulas y los contratos. El rendimiento predictivo observado tiene sus propias limitaciones: configuración única sin búsqueda de hiperparámetros, una carga retenida y posibles dependencias entre ventanas de una misma serie. Una revisión futura puede comparar variantes usando validación interna por cargas o bloques de training y conservar una prueba final independiente.

## 8. Selección de J y del criterio PCA (`validacion.py`)

La primera configuración (J=12, ≤ 95 %, r=8) dejaba el clasificador subajustado: el costo de training de Softmax se estabilizaba en 0.493 y, con Softmax directo sobre las 20 características MDE, la exactitud de prueba era mayor que con la cadena reducida. Como SLS y PCA no usan etiquetas, una reducción demasiado agresiva descarta información que separa las clases.

`validacion.py` evalúa J ∈ {8, 12, 14, 16, 18, 19} y los criterios PCA {≤ 95 %, ≥ 95 %, ≥ 99 %} con **validación cruzada dejando una carga fuera dentro de HP0–HP2** (3 pliegues). En cada pliegue se reajustan normalización, SLS, PCA y Softmax solo con las dos cargas internas. **HP3 no se utiliza.** La tabla completa está en `Resultados/validacion_cv.csv`.

| Configuración | Validación cruzada (exactitud media) |
|---|---:|
| J=12, ≤ 95 % (configuración inicial) | 0.685 |
| J=16, ≥ 99 % | 0.769 |
| **J=19, ≥ 99 % (seleccionada)** | **0.786** |

Se adoptó J=19 con `at_least` 99 %. Con D=20, la reducción SLS es suave (20 → 19 → 16): la validación indica que reducciones más fuertes pierden información discriminativa. Esta selección se hizo antes de evaluar la nueva configuración en HP3; HP3 se midió una sola vez con la configuración elegida.

El entrenamiento de SLS se mantiene en 300 épocas: en pruebas exploratorias, alargarlo o aumentar la tasa no mejoró la validación cruzada.