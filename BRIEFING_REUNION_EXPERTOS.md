# Briefing para la reunión con expertos — Proyecto CervixAI

> Documento para prepararte. Lenguaje simple a propósito. Al final hay un **glosario**
> con cada término técnico por si alguien lo suelta y no quieres perder el hilo.

---

## 0. Lo esencial en 30 segundos (para arrancar)

- **Qué es:** un sistema que **mide sola** la longitud del cuello del útero en una ecografía
  vaginal. Esa medida sirve para estimar el **riesgo de parto prematuro** (un cuello corto = más riesgo).
- **Cómo vamos:** la medición **ya funciona y ya da milímetros**. El modelo de inteligencia
  artificial reconoce **muy bien** las imágenes parecidas a las que se usó para enseñarlo, pero
  **falla con imágenes de otros ecógrafos**.
- **Qué necesito de la reunión:** (1) que los **médicos validen** que medimos bien; (2) que los
  **etiquetadores** nos ayuden a preparar imágenes de varios ecógrafos; (3) **permiso** para usar
  imágenes reales de forma segura y legal.

---

## 1. Qué hace el sistema (explicado facilito)

En la ecografía, el cuello del útero se ve como **dos "labios"** (uno arriba y uno abajo, como los
labios de una boca cerrada). El **canal** por donde se mediría es **la línea donde esos dos labios se
juntan**. Medir la "longitud cervical" es medir **el largo de esa línea, siguiendo su curva**, desde
un extremo (el interno, hacia el bebé) hasta el otro (el externo, hacia la vagina).

El sistema hace exactamente eso, en automático.

---

## 2. Cómo funciona por dentro (4 pasos)

1. **Reconocer los dos labios** — una IA llamada *U-Net* "colorea" en la imagen el labio de arriba y
   el de abajo. (A esto se le dice *segmentar*.)
2. **Encontrar el canal** — con geometría, trazamos la **línea que va justo en medio** de los dos
   labios, de punta a punta. Esa línea es el eje del canal.
3. **Medir siguiendo la curva** — el cuello casi nunca es recto; suele estar curvado. Así que medimos
   el **largo de la curva** (no una línea recta de punta a punta, que daría de menos).
4. **Pasar a milímetros** — la imagen médica (archivo *DICOM*) trae guardado **cuántos milímetros
   mide cada pixel**. Con ese dato convertimos el largo en píxeles a **milímetros reales**.

---

## 3. Qué YA funciona y qué NO (estado honesto)

| | Estado |
|---|---|
| Reconocer los labios (imágenes tipo "las de entrenamiento") | ✅ Excelente (calidad 0.96 de 1.0) |
| Encontrar el canal y medir su curva | ✅ Funciona en 440/440 imágenes de prueba |
| Convertir a milímetros con el DICOM | ✅ Funciona |
| Reconocer los labios en **imágenes de otros ecógrafos** | ❌ **Falla** (este es EL problema) |
| Validación **médica** de que la medida es clínicamente correcta | ⚠️ Falta (es para esta reunión) |
| Avisar "no confío en esta imagen" cuando algo sale raro | ⚠️ Falta (planeado) |

**Herramientas que ya dejamos listas:**
- Un script para **mejorar el modelo** (reentrenarlo para que aguante otros ecógrafos).
- Un script para **comparar** el modelo viejo vs. el mejorado **sin que las imágenes salgan de la
  computadora** (importante para la parte legal).

---

## 4. El problema central: "funciona con lo que conoce, falla con lo nuevo"

El término técnico es **fuera de distribución** (OOD). En cristiano:

> El modelo aprendió con imágenes de **un** tipo de ecógrafo (un set público). Con imágenes parecidas
> lo hace casi perfecto. Pero cada ecógrafo (GE, Voluson, Accuvix, Samsung…) produce imágenes con
> **brillo, contraste y textura distintos**, y ahí el modelo se confunde.

Lo importante que hay que entender (y que decidí después de **medirlo**, no de suponerlo):

- **El modelo NO está mal hecho.** En su terreno es excelente (mejor incluso que el modelo del paper
  de referencia).
- **El problema es de DATOS, no de "cambiar a una IA más avanzada".** Ninguna IA más sofisticada
  arregla esto sola: **lo único que lo arregla es enseñarle con imágenes de los ecógrafos reales**
  que se van a usar.

Por eso esta reunión importa tanto: **necesitamos datos representativos y validación clínica.**

---

## 5. Qué preguntar y pedir a CADA experto (lo más útil de este doc)

### 5A. A los MÉDICOS / especialistas del tema
El objetivo es que validen el **método** y nos den los criterios clínicos.

- ¿La forma correcta de medir es **siguiendo la curva** del canal (no en línea recta)? ¿Cuándo se usa
  recta y cuándo curva?
- ¿Cómo definen **exactamente** los dos extremos: el **orificio interno (OI)** y el **externo (OE)**?
  (Necesitamos saber dónde empieza y termina la medida.)
- ¿Cuál es el **margen de error aceptable** en milímetros para que sirva clínicamente? (¿1 mm? ¿2 mm?)
- ¿Qué **valor de corte** importa? (Por ejemplo: ¿menos de 25 mm = riesgo alto? Que ellos lo definan.)
- ¿Qué hace que una imagen sea **medible o no**? (plano correcto, vejiga vacía, sin presión de la sonda…)
- **Casos especiales** que debemos contemplar: cuello corto, *funneling* (embudo), moco/*sludge*, etc.
- ¿Tienen un **estándar de oro** o guía oficial que sigan? (p. ej. las guías **ISUOG**.)

### 5B. A los ESPECIALISTAS EN ETIQUETADO
El objetivo es conseguir **datos etiquetados** de varios ecógrafos para mejorar el modelo.

- Necesitamos que marquen, en cada imagen, **el labio anterior y el posterior por separado** (son las
  dos "clases" que aprende el modelo). Formato: máscaras PNG con valores {0 fondo, 1 anterior, 2 posterior},
  igual que el dataset actual. (Herramienta tipo *makesense.ai* sirve.)
- **MUY IMPORTANTE:** las imágenes para entrenar deben estar **limpias, SIN las cruces ni líneas de
  medición** que a veces deja el ecografista. Si el modelo ve esas marcas, aprende a "buscar las
  marcas" en vez de la anatomía, y luego falla con imágenes limpias. (Si solo hay imágenes con marcas,
  se pueden **borrar digitalmente**.)
- **Consistencia:** que etiqueten **2 personas** y se pongan de acuerdo cuando difieran (como hacen los
  papers). Eso sube mucho la calidad.
- **Diversidad:** que las imágenes cubran **los distintos ecógrafos** que realmente se usan.
- **Cantidad:** aunque sean **50–100 imágenes bien etiquetadas por ecógrafo**, ya ayuda a mejorar.
- ¿Con qué **herramienta** etiquetan y en qué **formato** exportan? (para que encaje con nuestro código.)

### 5C. A los EXPERTOS EN CÓDIGO / ingeniería
El objetivo es revisar arquitectura, entrenamiento e infraestructura.

- Revisión del **diseño del pipeline** (está modular, por etapas, con pruebas).
- La **estrategia contra el problema OOD**: ¿reentrenar con *augmentation* (ya lo dejé armado),
  juntar datos nuevos, o ambos?
- **Infraestructura de entrenamiento**: necesitamos **GPU** (una laptop normal tarda ~15 h; en una
  GPU son minutos). ¿Usamos Colab, un servidor, la nube?
- **Dónde vivirá el sistema** en producción (¿integrado al PACS del hospital? ¿app aparte?).
- **Seguridad y gobierno de datos**: cómo guardar/procesar los DICOM confidenciales de forma legal.
- Revisión de la **lógica de medición** (la parte geométrica del canal).

---

## 6. El permiso para usar las imágenes (cómo plantearlo)

Esto es delicado porque los datos son **confidenciales de pacientes**. La forma de pedirlo que da
más confianza:

**Lo que pedimos:** permiso para usar una **muestra de imágenes anonimizadas** (sin datos del
paciente) para dos cosas: (1) **medir** qué tan bien funciona el sistema con sus imágenes, y
(2) **mejorar** el modelo entrenándolo con ellas.

**Argumentos que tranquilizan (y son verdad, ya lo dejamos montado así):**
- **Las imágenes NO tienen que salir de sus instalaciones.** Todo el procesamiento corre **en local**;
  el sistema solo devuelve **números** (estadísticas), nunca las imágenes.
- El **etiquetado lo pueden hacer sus propios especialistas**, así los datos crudos nunca se comparten
  con nadie externo.
- Podemos trabajar solo con **imágenes anonimizadas** (quitando nombre, fecha, ID del paciente).

**Preguntas concretas para la parte médico-legal:**
- ¿Qué permite **el contrato y la ley** (protección de datos) exactamente?
- ¿Pueden ceder una muestra **anonimizada** para investigación/mejora del modelo?
- ¿El trabajo debe ser **100% en sitio** (on-site) o se puede sacar data anonimizada?
- ¿Quién firma/autoriza y qué documento necesitamos?

---

## 7. Preguntas que TE pueden hacer (y cómo contestar sin trabarte)

- **"¿Qué tan preciso es?"** → En imágenes de su tipo de entrenamiento, la segmentación es excelente
  (0.96). La medida en mm ya funciona; **falta que ustedes (médicos) validen el margen de error
  clínico**. En otros ecógrafos aún no es confiable — por eso pedimos datos.
- **"¿Ya lo probaron con nuestras imágenes?"** → Todavía no, porque son confidenciales. Justo por eso
  traemos una forma de probarlo **sin que las imágenes salgan de aquí** (solo salen números).
- **"¿Por qué no usan una IA más avanzada / el último modelo?"** → Lo evaluamos (el paper CL-Net). El
  problema **no es la IA, es tener datos de los ecógrafos reales**. Una IA más compleja tendría el
  mismo problema y costaría mucho más trabajo.
- **"¿Cuánto trabajo de etiquetado necesitan?"** → Con 50–100 imágenes bien etiquetadas por ecógrafo
  ya empezamos a mejorar; más, mejor.

---

## 8. Chuleta rápida (glosario + números + papers)

### Glosario (por si sueltan el término)
- **Ecografía transvaginal (TVUS):** el ultrasonido con la sonda interna; da la mejor vista del cuello.
- **Cuello uterino / cérvix:** la parte baja del útero que se mide.
- **Labio anterior / posterior:** las dos partes del cuello (arriba/abajo) entre las que va el canal.
- **Canal endocervical:** el conducto interno del cuello; su largo es lo que medimos.
- **OI (orificio interno) / OE (orificio externo):** los dos extremos de la medida.
- **Longitud cervical (CL):** la medida final, en mm. Corta = más riesgo de parto prematuro.
- **Segmentar / segmentación:** que la IA marque qué pixeles son cada estructura.
- **U-Net:** el tipo de red neuronal que segmenta (colorea) los labios. La que ya tenemos.
- **CL-Net:** el modelo del paper de referencia; enfoque más avanzado, para el futuro.
- **YOLO:** otro modelo de IA; en este proyecto también detecta/segmenta, es alternativa/apoyo.
- **DICOM:** el formato de archivo médico; guarda la imagen **y** datos como el tamaño real del pixel.
- **PixelSpacing:** el dato del DICOM que dice cuántos mm mide cada pixel (para pasar a milímetros).
- **Dice:** nota de 0 a 1 de qué tan bien coincide lo que marca la IA con lo correcto (1 = perfecto).
- **OOD (fuera de distribución):** imágenes distintas a las de entrenamiento; donde el modelo falla.
- **Augmentation (aumento de datos):** enseñarle al modelo versiones con distinto brillo/contraste/ruido
  para que aguante otros ecógrafos.
- **Fine-tune (afinado):** reentrenar un poco un modelo ya hecho, sin empezar de cero.
- **Ground truth (verdad de referencia):** lo que un experto marcó a mano; el "correcto" contra el que
  comparamos.
- **ISUOG:** sociedad internacional cuyas guías estandarizan cómo medir en ecografía obstétrica.

### Números clave que puedes citar
- Calidad de segmentación en su terreno: **Dice 0.96** (excelente; el paper CL-Net reporta 0.92).
- Cobertura de medición: **440/440** imágenes de prueba dan una medida válida.
- Calibración: confirmado que los DICOM traen `PixelSpacing` (ej. **0.088 mm/pixel**).

### Los 3 papers que respaldan el enfoque
- **CL-Net (Kwon, 2024)** — el más parecido a lo nuestro: mide longitud cervical con IA, validado
  contra médicos (error ~1.3 mm). Confirma medir **siguiendo la curva** y calibrar con **metadata**.
- **O-CCR (2025)** — confirma que el canal es la **línea donde se tocan los labios**, no un hueco.
- **Włodarczyk (2020)** — técnica para **borrar las marcas** de medición de las imágenes.

---

### Cierre: las 3 cosas que quieres salir con ellas de la reunión
1. **Médicos:** visto bueno del método + criterios (extremos OI/OE, margen de error, valor de corte).
2. **Etiquetadores:** un plan para **etiquetar imágenes de varios ecógrafos** (limpias, 2 revisores).
3. **Legal/dirección:** **permiso** para usar una muestra anonimizada, aclarando si es solo en sitio.
