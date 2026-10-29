# Evaluación 2 — implementación concluida

Directorio de trabajo: `C:\Users\matia\Documents\AID\Eval2-Implementacion`.
Fecha: 29 de octubre de 2026. Código, datos procesados, modelos, documentación y resultados están en `Entregables`.

## Checklist de los doce pasos

- [x] **1. Fuentes:** enunciado y apuntes conservados en el proyecto; fórmulas y páginas identificadas en `Entregables/METODOLOGIA.md`.
- [x] **2. Decisiones:** coseno, DE normalizada, ventanas, escalas, cargas, normalización, semillas y optimización fijados en `configuracion.json`; ambigüedades documentadas.
- [x] **3. Datos:** cuatro CSV de 120000 × 4 revisados; clases y cargas tomadas de `FormatData.pdf`; originales idénticos a `DATA.zip`.
- [x] **4. MDE:** extracción completa; dos ejemplos de entropía del PDF reproducidos; escalas, retardo, límites y series constantes comprobados.
- [x] **5. Preparación:** 928 ventanas, shuffle conjunto, etiquetas one-hot, procedencia guardada; 696 training y 232 testing; estadísticas aprendidas solo con training.
- [x] **6. SLS:** coseno, RLS y RMSprop implementados; gradiente comprobado por diferencias finitas; pesos y reconstrucción guardados.
- [x] **7. PCA:** SVD sobre representación centrada, selección literal ≤ 95 %, media y proyección persistidas; ocho componentes con 92.6582 % de varianza.
- [x] **8. Softmax:** CE media, sesgo, RMSprop, historial y gráfica; archivo exacto `pesoftmax`; época guardada identificada.
- [x] **9. Test:** modelos recargados sin reajuste; matriz de confusión, F-scores y predicciones exportados; filas predichas y columnas reales.
- [x] **10. Verificación:** 26 pruebas correctas, sin errores ni omisiones; identidad de transformaciones, gradientes, costos, métricas y persistencia comprobadas.
- [x] **11. Entregables:** cuatro scripts exigidos, módulo auxiliar, siete archivos de datos/modelos, dependencias y documentación disponibles; revisión visual de las cuatro gráficas.
- [x] **12. Integración:** cadena repetida desde otra carpeta con 15 artefactos idénticos byte a byte; inferencia en proceso nuevo; paquete ZIP comprobado.

## Resultado de la ejecución

| Indicador | Resultado |
|---|---:|
| Dimensiones MDE → SLS → PCA | 20 → 12 → 8 |
| Ventanas de entrenamiento / prueba | 696 / 232 |
| Cargas de entrenamiento / prueba | HP0–HP2 / HP3 |
| Aciertos en prueba | 165 de 232 |
| Exactitud | 71.1207 % |
| F1 macro | 0.703450 |
| CE de training del Softmax guardado | 0.493002 |

La mayor confusión ocurre entre daño en elementos rodantes y estado saludable: 36 ventanas de esa falla se clasificaron como saludables. El resultado mide transferencia a HP3 dentro de los datos suministrados; no demuestra generalización a otros rodamientos. No se cambiaron hiperparámetros a partir de este resultado.

## Dónde revisar

- **Ejecución:** `Entregables/README.md` contiene instalación, comandos, formatos y resultados por clase.
- **Matemáticas:** `Entregables/METODOLOGIA.md` explica cada fórmula y las adaptaciones de extremos, orientación matricial y normalización sin fuga de información.
- **Código:** `Entregables/mdef.py`, `dimr.py`, `train.py`, `test.py` y el auxiliar `comun.py`.
- **Resultados:** `Entregables/Resultados`, con las tablas, cuatro gráficas, probabilidades y manifiesto SHA-256.
- **Evidencia:** `Trabajo/registros/verificacion_final.json`, `pruebas_matematicas.log`, `fuentes.json`, `procedencia.csv` y registros por etapa.
- **Pruebas repetibles:** `Trabajo/verificaciones/test_matematicas.py` y `verificar_integracion.py`.
- **Entrega:** `Entregables.zip`; contiene únicamente `Entregables`, sin entorno virtual, datos originales ni experimentos.

`Plan-Evaluacion2.md` y su PDF se conservan como documentos previos. Los puntos que allí figuraban pendientes quedaron resueltos mediante los apuntes adicionales y las decisiones registradas en `METODOLOGIA.md`. La implementación mantiene los nombres exigidos y no añade extensiones a `dtrain`, `dtest`, `pesos`, `pesoftmax` ni `costo`.

Para reproducir toda la cadena se necesita el proyecto con `data/originales`. Para evaluar el modelo ya guardado basta el contenido del ZIP y las dependencias instaladas. El paquete queda preparado para revisión; no se ha enviado al aula virtual.
