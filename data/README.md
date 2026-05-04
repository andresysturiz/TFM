# 📁 **Datasets utilizados en el TFM**

Este directorio contiene los datos empleados en el Trabajo Fin de Máster para comparar **Partial Least Squares (PLS)** con métodos clásicos de regresresión (OLS, Ridge, Lasso, PCR) en tres escenarios complementarios:

- **Quimiometría con espectros NIR** (Tecator, Gasoline)  
- **Alta dimensionalidad p≫n** (Riboflavin)

La estructura del directorio es:

```
data/
 ├── raw/         # Datos originales sin modificar
 └── processed/   # Datos generados tras preprocesado (escalado, splits, etc.)
```

---

# 🍖 1. Tecator Dataset

**Fuente:** StatLib / UCI (versión clásica usada en quimiometría)  
**Instancias:** 215  
**Variables:** 100 absorbancias NIR + 3 variables químicas  
**Tarea:** Regresión  
**Valores faltantes:** No  

### 📌 Descripción
Dataset clásico de espectros NIR de muestras de carne.  
El objetivo es predecir el contenido de **grasa (`fat`)** a partir de 100 absorbancias medidas entre 850–1050 nm.

### 🔬 Variables

**Predictoras (100 absorbancias):**  
`absorbance_1`, `absorbance_2`, …, `absorbance_100`

**Objetivo:**  
- **fat** (contenido graso)

### 📂 Archivo
```
data/raw/tecator.csv
```

---

# ⛽ 2. Gasoline Dataset

**Fuente:** NIR spectroscopy benchmark dataset  
**Instancias:** 60  
**Variables:** 401 absorbancias NIR + 1 objetivo  
**Tarea:** Regresión  
**Valores faltantes:** No  

### 📌 Descripción
Dataset de espectros NIR de gasolina.  
El objetivo es predecir el **octanaje** a partir de 401 absorbancias.

### 🔬 Variables

**Predictoras:**  
`nir_1`, `nir_2`, …, `nir_401`

**Objetivo:**  
- **octane**

### 📂 Archivo
```
data/raw/gasoline.csv
```

---

# 🧬 3. Riboflavin Dataset (p≫n)

**Fuente:** Bühlmann et al. (High-Dimensional Statistics)  
**Instancias:** 71  
**Variables:** 4088 genes + 1 objetivo  
**Tarea:** Regresión p≫n  
**Valores faltantes:** No  

### 📌 Descripción
Dataset de expresión génica con **alta dimensionalidad** (p≫n).  
El objetivo es predecir la **producción de riboflavina** a partir de 4088 genes.

### 🔬 Variables

**Predictoras:**  
Genes con nombres tipo:  
`AADK_at`, `AAPA_at`, `ABFA_at`, … (4088 en total)

**Objetivo:**  
- **x** (producción de riboflavina)

### 📂 Archivo
```
data/raw/riboflavin.csv
```

---

# 🛠️ 4. Datos procesados

Los archivos generados tras preprocesado (escalado, normalización, splits train/test, etc.) se almacenan en:

```
data/processed/
```

Estos se generan automáticamente al ejecutar:

```bash
python -m repo_tfm
```

---

# 🔄 5. Reproducibilidad

Todos los datasets incluidos son **públicos** y se distribuyen únicamente con fines académicos para la reproducibilidad del TFM.

---

# 📜 6. Licencia y uso

Los datasets provienen de fuentes públicas (StatLib, NIR benchmark datasets, literatura científica).  
Su uso en este repositorio es exclusivamente académico.
