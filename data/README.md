# 📁 Datasets utilizados en el TFM

Este directorio contiene los conjuntos de datos empleados en el Trabajo Fin de Máster dedicado al análisis de la regresión por mínimos cuadrados parciales (**PLS**) y su comparación con métodos clásicos de regresión y regularización (**OLS, Ridge, Lasso y PCR**).

Los datasets fueron seleccionados para representar distintos escenarios relevantes en regresión multivariante:

* 🔗 Multicolinealidad severa
* 🧩 Presencia de estructura latente
* 📈 Problemas de alta dimensionalidad ((p \gg n))
* ⚙️ Contextos típicos de quimiometría y bioinformática

---

# 📊 Resumen de datasets

| Dataset       | Observaciones ((n)) | Variables ((p)) | Dominio            | Características principales            |
| ------------- | ------------------: | --------------: | ------------------ | -------------------------------------- |
| 🍖 Tecator    |                 215 |             100 | Espectroscopía NIR | Multicolinealidad y estructura latente |
| ⛽ Gasoline    |                  60 |             401 | Quimiometría       | Alta correlación espectral             |
| 🧬 Riboflavin |                  71 |            4088 | Expresión génica   | Alta dimensionalidad ((p \gg n))       |

---

# 🗂️ Estructura del directorio

```text
data/
├── raw/         # Datos originales sin modificar
└── processed/   # Datos generados tras el preprocesamiento
```

Los datos procesados incluyen:

* 📏 Estandarización
* ✂️ Particiones train/test
* 🔄 Transformaciones de matrices
* 📦 Estructuras auxiliares para validación y evaluación

---

# 🍖 1. Tecator Dataset

**Fuente:** StatLib / UCI Repository
**Tipo de problema:** Regresión

## 📌 Descripción

El dataset *Tecator* contiene espectros de absorción en infrarrojo cercano (**NIR**) obtenidos a partir de muestras de carne. Cada observación incluye 100 absorbancias medidas en distintas longitudes de onda.

El objetivo consiste en predecir el contenido de grasa (`fat`) a partir de las señales espectrales.

Este dataset constituye un caso clásico de quimiometría caracterizado por:

* 🔗 Multicolinealidad severa
* 🌊 Alta correlación entre longitudes de onda adyacentes
* 🧩 Presencia de estructura latente

## 🔬 Variables

### Predictoras

```text
absorbance_1 ... absorbance_100
```

### Variable objetivo

```text
fat
```

## 📂 Archivo

```text
data/raw/tecator.csv
```

---

# ⛽ 2. Gasoline Dataset

**Fuente:** NIR Spectroscopy Benchmark Dataset
**Tipo de problema:** Regresión

## 📌 Descripción

El dataset *Gasoline* contiene espectros NIR de muestras de gasolina registrados en 401 longitudes de onda.

El objetivo es predecir el índice de octanaje (`octane`) a partir de las absorbancias espectrales.

Este conjunto representa un escenario de:

* 📈 Alta dimensionalidad
* 🔗 Fuerte multicolinealidad
* 🧩 Estructura latente asociada a propiedades químicas de la mezcla

## 🔬 Variables

### Predictoras

```text
nir_1 ... nir_401
```

### Variable objetivo

```text
octane
```

## 📂 Archivo

```text
data/raw/gasoline.csv
```

---

# 🧬 3. Riboflavin Dataset

**Fuente:** Bühlmann et al. — *High-Dimensional Statistics*
**Tipo de problema:** Regresión (p \gg n)

## 📌 Descripción

El dataset *Riboflavin* corresponde a un estudio de expresión génica en *Bacillus subtilis*.

Cada observación contiene mediciones de expresión para 4088 genes, mientras que el objetivo consiste en predecir la producción de riboflavina.

Este conjunto constituye un escenario clásico de:

* 🚀 Alta dimensionalidad ((p \gg n))
* 🔗 Multicolinealidad extrema
* 🧬 Coexpresión génica
* ✨ Sparsity (solo un subconjunto reducido de genes resulta relevante)

## 🔬 Variables

### Predictoras

Genes con identificadores del tipo:

```text
AADK_at, AAPA_at, ABFA_at, ...
```

### Variable objetivo

```text
x
```

## 📂 Archivo

```text
data/raw/riboflavin.csv
```

---

# ⚙️ Preprocesamiento y reproducibilidad

Los datos procesados se generan automáticamente mediante el pipeline experimental implementado en el proyecto:

```bash
python -m repo_tfm
```

El procedimiento incluye:

* 📏 Estandarización
* ✂️ Particiones train/test
* 🔄 Preparación de matrices
* 📊 Generación de estructuras auxiliares para validación y evaluación

Para garantizar reproducibilidad:

* 🔒 Todos los datasets utilizados son públicos
* 🎲 Se controlan las semillas aleatorias (`random_state`)
* ⚖️ Todos los modelos se ejecutan bajo un pipeline homogéneo

---

# 📜 Licencia y uso

Todos los datasets provienen de fuentes públicas y se incluyen exclusivamente con fines académicos y de investigación.
