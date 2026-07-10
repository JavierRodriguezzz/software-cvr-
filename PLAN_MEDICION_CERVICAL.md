# Plan técnico — Medición automática de longitud cervical

> Documento de diseño para revisión. **No hay código escrito aún.** Basado en el audit del
> pipeline actual y en tres papers de referencia (CL-Net / Kwon 2024, O-CCR / Hwangbo 2025,
> Włodarczyk 2020). Objetivo: que el sistema mida de forma automática la longitud del canal
> cervical sobre imágenes limpias, en milímetros, y que podamos validar esa medición contra
> el ground truth del experto.

---

## 0. Objetivo y caso de uso

**Caso A confirmado:** el sistema recibe una imagen ecográfica **limpia** (sin cruces de caliper)
y debe producir, de forma automática:

1. La segmentación de los labios cervicales (anterior / posterior) — *ya funciona*.
2. El **eje del canal cervical** de OI (orificio interno) a OE (orificio externo).
3. La **longitud de arco** de ese eje (no la distancia recta, porque el cuello suele estar curvado).
4. La conversión a **milímetros** usando la escala del DICOM.
5. Un **puntaje de confianza** de la medición.

Todo esto se contrasta con las imágenes que marcó el experto (cruces + línea), usadas solo como
**referencia visual para revisión humana** — nunca entran como input al modelo ni se procesan por código.

---

## 1. Estado actual (resumen del audit)

| Componente | Estado | Nota |
|---|---|---|
| Segmentación de labios (U-Net, `models/unet_model.py`) | ✅ Funciona | 3 clases {fondo, labio ant.=1, labio post.=2}, a resolución original |
| Segmentación de labios (YOLO, `yolo_cervix_best.pt`) | ✅ Funciona | Alternativa/redundante a la U-Net |
| Preprocesamiento (CLAHE+denoise, `core/preprocessing/`) | ⚠️ Código muerto | Se calcula pero no se usa: `SegmentationStage` pasa `original_image` cruda |
| **Extracción del canal** (`core/measurement/canal_extractor.py`) | ❌ **Roto** | "Dilatar hasta que se toquen" → 1-4 px. Enfoque conceptualmente equivocado |
| Esqueleto / centerline (`skeleton_extractor.py`, `centerline_extractor.py`) | ❌ Roto | (a) esqueleto vacío por el bug anterior; (b) mismatch de firma en `measurement_engine.py:127` |
| Calibración (`core/calibration/calibration_engine.py`) | ❌ Falla 12/12 | Busca barra de escala visual; nunca lee el DICOM |
| Confianza (`core/confidence/`) | 🟡 Parcial | Funciona pero `mean_probability` siempre = 0.5 (la U-Net no expone probabilidades) |
| Visualización (`core/visualization/`) | ✅ OK | Genera overlays por capa |
| Tests `tests/` (7 de 8) | ❌ No importan | Referencian módulos de una arquitectura vieja inexistente |
| Tests `core/tests/` | ✅ 22 pasan | Cobertura parcial |

Detalle completo en la memoria del proyecto (`project-cervixai-pipeline-audit`).

---

## 2. Arquitectura objetivo

```
Imagen limpia (PNG/JPG)  ──┐
DICOM (.dcm) ──────────────┤
                           ▼
        [0] Carga + (opcional) recorte ROI
                           ▼
        [1] Preprocesamiento  ← decidir: arreglar o eliminar
                           ▼
        [2] Segmentación labios (U-Net)  ✅ existe
                           ▼
        [3] CANAL = línea media entre labio ant. y post.  ← EL FIX
            (watershed / frontera de Voronoi, no "hueco")
                           ▼
        [4] OI/OE = extremos del eje  +  longitud de arco (polilínea)
                           ▼
        [5] Calibración: mm/píxel desde DICOM  →  longitud en mm
                           ▼
        [6] Confianza
                           ▼
        [7] Visualización + payload JSON
                           ▼
        [V] Validación contra ground truth (cruces del experto)
```

Comparado con CL-Net: nuestra etapa [2] equivale a su segmentación auxiliar de labios; la etapa
[3]-[4] reemplaza su "cabeza de heatmap del canal" por una construcción **geométrica** (sin
reentrenar). Si la validación muestra que no basta, la sección 8 describe cómo escalar a la cabeza
de heatmap real de CL-Net.

---

## 3. Activos de datos y su rol

| Activo | Ubicación | Rol |
|---|---|---|
| Imágenes limpias + máscaras de labios (**dataset público**) | `models/unet_dataset/` (352/44/44) | Con ellas se entrenaron U-Net y YOLO. Sin marcar (GT aparte). **Input** de producción y de las pruebas locales de desarrollo |
| Imágenes marcadas del asesor (**confidenciales**) — JPG/PNG **+ DICOM** | Local (no compartidas, no usadas para entrenar ni tocadas) | Cruces + línea = **referencia visual** de OI/OE/canal; el DICOM aporta `PixelSpacing`. Validación en el entorno del usuario |
| Etiquetas de recuadro (makesense, YOLO) | Local (privado) | (Opcional) etapa de localización |

**Nota importante:** los dos sets son de **origen distinto**. El limpio es público (para entrenar y
para probar el canal en local, pero **sin mm** porque no tenemos su DICOM). El marcado es del asesor,
confidencial, y es el único con DICOM → la validación con milímetros y contra las marcas del experto
ocurre en el entorno del usuario, no aquí.

**Regla dura:** el modelo solo ve imágenes limpias. Las imágenes con cruces se usan como
**referencia visual para revisión humana** (comparar a ojo si el canal automático coincide con el
que marcó el experto); no se procesan ni se extraen por código.

---

## 4. Especificación etapa por etapa

### 4.1 Preprocesamiento — *decisión pendiente*

Hoy `PreprocessingStage` calcula CLAHE + denoise pero la U-Net lo ignora (hace su propio
resize/gris/normalize en `models/unet_model.py::predict`). Dos opciones:

- **(a) Eliminar** la etapa de preprocesamiento del camino de inferencia (más simple, honesto con
  lo que realmente pasa). El modelo ya fue entrenado sin CLAHE, así que meterlo ahora podría incluso
  empeorar la segmentación.
- **(b) Conservarla** solo para la etapa de calibración/validación si allí ayuda.

**Recomendación:** (a) — quitarla del path de segmentación para no gastar ~160 ms por corrida en algo
que no se usa. Mantener el `gray_image` que sí consumen otras etapas.

### 4.2 Segmentación de labios — *mantener*

Sin cambios. `UNetSegmenter.predict` devuelve la máscara {0,1,2} a **resolución original** (hace
resize de vuelta con `INTER_NEAREST`), lo cual es clave: la medición y la calibración ocurren en el
espacio de píxeles original, que es al que aplica el `PixelSpacing` del DICOM.

*Mejora opcional:* exponer el mapa de probabilidad (softmax) de la U-Net para que la confianza
(`mean_probability`) deje de ser un 0.5 fijo. Requiere tocar `models/unet_model.py`. Baja prioridad.

### 4.3 Extracción del canal — **el arreglo central**

**Reemplazar por completo** `CanalGapExtractor` (dilatar-hasta-tocar). El canal es la **línea media
entre el labio anterior y el posterior** — la curva equidistante de ambas máscaras. Método propuesto
(geométrico, sin ML, robusto aunque los labios se toquen):

1. Tomar `mask_anterior` (A) y `mask_posterior` (P) de la etapa de limpieza.
2. Construir marcadores: A → etiqueta 1, P → etiqueta 2.
3. Calcular la **frontera de separación** entre A y P mediante *watershed* / SKIZ (skeleton by
   influence zones = frontera de Voronoi entre las dos regiones). Herramientas: `scipy.ndimage`
   (distance transform) + `skimage.segmentation.watershed`. La línea de watershed entre las dos
   cuencas es una curva de 1 px que recorre justo el medio del espacio entre los labios.
4. **Recortar** esa curva a la zona anatómicamente relevante: intersectarla con una dilatación de
   `A ∪ P` (para que no se extienda hacia los bordes de la imagen) y quedarnos con la **componente
   conexa más larga**.
5. Resultado: el eje del canal como una polilínea ordenada de un extremo a otro.

*Por qué funciona donde el método viejo falla:* cuando los labios se tocan (caso frecuente, según
O-CCR el canal aparece como una línea ecogénica en el contacto), la frontera de watershed pasa
exactamente por ese contacto — que es justo el canal. No necesita que exista un "hueco vacío".

*Riesgo / a validar:* la extracción exacta de la cresta y su recorte a la zona correcta es la parte
que hay que afinar visualmente sobre imágenes reales. Es donde pondremos más iteración.

### 4.4 OI/OE + longitud de arco

Sobre la polilínea del canal:

1. **Extremos = OI y OE.** Los dos puntos terminales de la curva. Para distinguir cuál es cuál,
   usar una convención geométrica (p. ej. el extremo más cercano al fondo uterino = OI) y **confirmarla
   visualmente** con las imágenes marcadas por el experto. No es crítico para la *longitud*,
   solo para el etiquetado.
2. **Suavizado:** ajustar un B-spline (ya existe `SplineMeasurement`) para quitar el ruido de
   escalera del watershed.
3. **Longitud de arco:** sumar los segmentos de la polilínea (equivalente a los 8 segmentos de
   CL-Net; podemos remuestrear a N segmentos iguales). Esto da `arc_length_px`.

Reutilizamos gran parte de `MeasurementResult` y `SplineMeasurement`; hay que **corregir el
mismatch de firma** en `measurement_engine.py:127` y reconectar el flujo al nuevo extractor de canal
(el `SkeletonExtractor` viejo probablemente queda obsoleto o se repurposa).

### 4.5 Calibración por DICOM — *nuevo*

Confirmado: los DICOM traen el tag **`PixelSpacing`** (ejemplo real: `0.087975\0.087975` → 0.087975 mm
por píxel en fila y columna). **El valor es variable**: distintos conjuntos de imágenes tienen distinto
PixelSpacing, así que hay que leerlo por archivo en tiempo de ejecución, nunca hardcodearlo.

Orden de preferencia de la fuente de escala:

1. **DICOM `PixelSpacing`** (mm, confirmado presente). `pydicom` lo devuelve como `[fila, columna]`;
   usar ambos componentes (aquí son iguales = píxel cuadrado, pero lo dejamos general).
2. **DICOM `SequenceOfUltrasoundRegions[0].PhysicalDeltaX/Y`** como respaldo (suele venir en cm/px →
   convertir a mm) si algún DICOM no trajera `PixelSpacing`.
3. **Barra de escala visual** (el método actual) como *fallback* para PNG/JPG sin DICOM.
4. **Manual** (el usuario ingresa mm/píxel) como último recurso.

Puntos finos:
- El preview PNG que genera `dicom_to_preview` conserva la resolución del `pixel_array`, así que el
  `PixelSpacing` aplica directo a la imagen sobre la que medimos, **siempre que no haya resize**.
  Si en algún punto se redimensiona la imagen, hay que escalar el mm/píxel por el mismo factor.
- `app.py` ya lee el DICOM en `/upload`; hay que **extraer y guardar** el `PixelSpacing` en el
  `metadata` del archivo y pasarlo al `PipelineContext` para que `MeasurementStage` lo use.
- Para PNG/JPG sueltos (sin DICOM) no habrá mm — se reporta solo en px, o se usa el fallback.
- **No dispongo de DICOM reales** (confidenciales por contrato, no se comparten). La lógica de lectura
  se desarrolla y prueba contra un **DICOM sintético** generado con `pydicom` (con un `PixelSpacing`
  conocido como `0.087975\0.087975`); en tu entorno se valida con los archivos reales.

### 4.6 Confianza

Mantener `ConfidenceStage`. Con el canal ya funcionando, sus componentes (continuidad del esqueleto,
plausibilidad anatómica, consistencia geométrica) empezarán a dar valores reales. Revisar los rangos
del `config.yaml` una vez tengamos mediciones. Opcional: alimentar `mean_probability` con el softmax
real (ver 4.2).

### 4.7 (Opcional, fase posterior) Localización YOLO / orientación

- El recuadro de makesense sirve como **etapa 1 de recorte** (enfocar el ROI antes de segmentar,
  como la etapa 1 de CL-Net). No es imprescindible: los labios ya segmentan bien en la imagen completa.
- **Ojo con la orientación:** el recuadro de makesense es *recto* y no captura el ángulo del cuello.
  Para "orientarse según la línea" hay dos caminos: (a) cajas **rotadas** estilo O-CCR (x,y,w,h,θ),
  que implican re-etiquetar/entrenar; o (b) derivar la orientación *después*, de la propia curva del
  canal (más simple, sin modelo extra). **Recomendación:** (b), y dejar el YOLO-box solo si el recorte
  mejora la robustez medida.

---

## 5. Validación (revisión humana, sin extracción de datos)

Como las imágenes con cruces son solo **referencia visual** (no se procesan por código) y los DICOM
son confidenciales, la validación es **humana / visual**, no automática por extracción de color:

1. El pipeline genera un overlay del canal automático (eje + OI/OE) sobre la imagen limpia.
2. Se coloca lado a lado con la imagen equivalente marcada por el experto (cruces + línea).
3. Una persona del equipo revisa si el eje automático coincide con el trazo del experto y si OI/OE
   caen donde corresponde.
4. Opcional: para un puñado de casos se puede anotar **a mano** la longitud de referencia (leída de la
   imagen marcada) y compararla con la longitud automática en mm, para tener una idea del error.

Esto es menos riguroso que las métricas automáticas de los papers, pero respeta tus restricciones
(confidencialidad y no tocar las marcas por código). Si más adelante quieres métricas cuantitativas,
habría que **anotar manualmente** un set de referencia — no extraer las cruces.

---

## 6. Archivos a tocar

| Archivo | Acción |
|---|---|
| `core/measurement/canal_extractor.py` | **Reescribir**: watershed/Voronoi entre labios en vez de dilatar-hasta-tocar |
| `core/measurement/measurement_engine.py` | Corregir firma (`:127`), reconectar al nuevo extractor, remuestreo N-segmentos |
| `core/measurement/centerline_extractor.py` | Reconciliar/simplificar (unificar la firma; puede absorber el nuevo método) |
| `core/measurement/skeleton_extractor.py` | Probablemente **obsoleto** con el nuevo enfoque; evaluar quitar |
| `core/calibration/calibration_engine.py` | Añadir fuente DICOM (PhysicalDelta/PixelSpacing); barra de escala pasa a fallback |
| `core/preprocessing/preprocessor.py` + `pipeline_factory.py` | Sacar el preprocesamiento del path de inferencia (decisión 4.1a) |
| `app.py` | Leer y propagar mm/píxel del DICOM al `PipelineContext`; pasar metadata de escala |
| `core/config.py` + `config.yaml` | Nuevos parámetros (calibración DICOM, N segmentos, umbrales del canal) |
| `core/validation/` (nuevo) | Extractor de cruces (ground truth) + harness de métricas |
| `tests/` | Arreglar o eliminar los 7 archivos que no importan; añadir tests del nuevo canal |

---

## 7. Orden de trabajo por fases

**Fase 1 — Que mida (desbloquea el core). ✅ HECHO.**
Reescrito el extractor de canal como **banda equidistante entre labios** (`canal_extractor.py`,
clase `CanalCorridorExtractor`), corregidos los dos bugs de firma en `measurement_engine.py`,
corregido un bug de **conectividad-4** en `skeleton_extractor.py` (partía el eje diagonal en trozos
sub-umbral y podaba todo), e hrecho robusto el spline (`spline_measurement.py`: presupuesto de
suavizado proporcional + guarda anti-oscilación que revierte a la polilínea cruda).
Validado visualmente y por tasa de éxito: **de 0/12 a 44/44 (test) y 44/44 (val)**, con ejes curvos
anatómicamente correctos (mediana ~260-270 px). Pendiente menor: el eje se extiende ~30 px (config
`contact_zone_iters`) más allá de la punta del labio en el extremo OI ("cola"); refinar el recorte de
extremos al validar contra las marcas del experto.

**Fase 2 — Calibración DICOM (4.5). ✅ HECHO.**
`core/calibration/dicom_scale.py` lee `PixelSpacing` (→ `SequenceOfUltrasoundRegions` → `ImagerPixelSpacing`).
`CalibrationStage` lo consume vía `context.metadata["mm_per_pixel"]` (método `dicom_pixel_spacing`), y cae a
barra de escala para PNG/JPG. `app.py` lee la escala al subir el DICOM y la propaga al pipeline. Corregido
un **bug de orden**: la calibración corría *después* de la medición, así que `arc_length_mm` nunca se llenaba
→ reordenado calibración antes de medición. Verificado end-to-end (px × mm/px = mm) contra DICOM sintético
(pydicom) porque los reales son confidenciales. 6 tests nuevos en `core/tests/test_dicom_scale.py`.
→ Entregable: longitud en mm. **Logrado.**

**Fase 3 — Validación visual (5).**
Overlays del canal automático lado a lado con las imágenes marcadas por el experto; revisión humana.
→ Entregable: veredicto de si el eje/medición coincide. Punto de decisión: ¿basta el enfoque geométrico?

**Fase 4 — Limpiezas (9).**
Preprocesamiento muerto, tests rotos, legacy duplicado.

**Fase 5 — Opcionales / escalamiento.**
YOLO-box de localización (4.7) y/o cabeza de heatmap CL-Net (8), *solo si la Fase 3 lo justifica*.

---

## 8. Escalamiento a CL-Net (si las métricas de la Fase 3 no bastan)

Si la medición geométrica no alcanza la precisión del experto, el siguiente paso es el método
validado de Kwon 2024:
- Añadir a la U-Net una **segunda cabeza** que regrese un **heatmap de confianza del canal**
  (target = transformada de distancia `e^(−λ·dist)` a la curva del experto, λ=1).
- **Etiquetas de entrenamiento = la curva del canal**, que habría que **anotar manualmente** (no se
  extraen de las cruces por código). Es el mayor costo de este camino.
- Encoder compartido entre la segmentación de labios y la cabeza del canal (la clave de CL-Net).
- Extraer la cresta del heatmap → misma etapa [4] de arc length.

Esto es un proyecto de ML (reentrenar), por eso queda como escalón posterior y condicionado a datos.

---

## 9. Limpiezas técnicas

- ✅ **Hecho — código muerto eliminado:** `CervicalMeasurementPipeline` (en `models/unet_model.py`),
  `models/cervical_measurement_service.py` (archivo borrado), las clases huérfanas
  `CenterlineExtractionStage` y `SplineMeasurementStage`, y los accesores muertos `get_unet` /
  `get_cervical_service` en `app.py` (+ el `import torch` que ya no se usaba). Verificado: `app.py`
  importa, 22 tests pasan, el pipeline corre.
- **Pendiente — Preprocesamiento** (4.1a): decisión abierta (pregunta 1).
- **Pendiente — Tests rotos:** `tests/test_calibration.py`, `test_confidence.py`,
  `test_domain_models.py`, `test_pipeline_context.py`, `test_pipeline_manager.py`,
  `test_preprocessing.py`, `test_visualization.py` importan módulos inexistentes (arquitectura vieja)
  → reescribir contra la arquitectura real o borrar (decisión aparte).
- **Pendiente — `models/03_eval_cervical_length (1).py`:** script de evaluación legacy; revisar si
  sirve de referencia antes de borrar.

---

## 10. Preguntas abiertas / decisiones pendientes

1. **Preprocesamiento:** ¿confirmas eliminarlo del path de inferencia (4.1a)?
2. **YOLO-box:** ¿lo queremos como etapa de recorte, o derivamos orientación del propio canal (4.7b)?
3. **Reparto de validación (confirmar):** sets limpio (público, sin mm) y marcado (asesor, confidencial,
   con DICOM) son DISTINTOS → desarrollo/prueba visual del canal en local con las públicas; validación
   con mm + marcas del experto en tu entorno con las del asesor. ¿De acuerdo?
4. **Umbral clínico:** ¿reportamos algún corte de riesgo (p. ej. <25 mm) o solo el valor medido?
   (Kwon no fija corte; conviene definirlo con criterio clínico de tu equipo.)
```
