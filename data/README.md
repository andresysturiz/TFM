
# 📁 Datasets utilizados en el TFM

Este directorio contiene los datos empleados en el Trabajo Fin de Máster para comparar **Partial Least Squares (PLS)** con métodos clásicos de regresión en dos dominios distintos: química (Wine Quality) y aeroacústica (Airfoil Self‑Noise).

La estructura del directorio es:

```
data/
 ├── raw/         # Datos originales descargados sin modificar
 └── processed/   # Datos generados tras preprocesado (escalado, folds, etc.)
```

---

# 🍷 1. Wine Quality Dataset

**Fuente:** UCI Machine Learning Repository  
**ID:** 186  
**Referencia:** Cortez et al. (2009), *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems.  
**Instancias:** 4.898  
**Variables:** 11 predictoras + 1 objetivo  
**Tarea:** Regresión  
**Valores faltantes:** No

### 📌 Descripción
Dataset con muestras de vino verde portugués (tinto y blanco). El objetivo es predecir la **calidad sensorial** (0–10) a partir de propiedades fisicoquímicas medidas en laboratorio.

### 🔬 Variables

**Predictoras:**
- fixed_acidity  
- volatile_acidity  
- citric_acid  
- residual_sugar  
- chlorides  
- free_sulfur_dioxide  
- total_sulfur_dioxide  
- density  
- pH  
- sulphates  
- alcohol  

**Objetivo:**
- **quality** (puntuación sensorial entre 0 y 10)

### 📂 Archivo
```
data/raw/wine_quality.csv
```

---

# ✈️ 2. Airfoil Self‑Noise Dataset

**Fuente:** UCI Machine Learning Repository  
**ID:** 291  
**Procedencia:** NASA  
**Instancias:** 1.503  
**Variables:** 5 predictoras + 1 objetivo  
**Tarea:** Regresión  
**Valores faltantes:** No

### 📌 Descripción
Dataset obtenido de pruebas aeroacústicas con perfiles aerodinámicos **NACA 0012** en túnel de viento anecoico. El objetivo es predecir el **nivel de presión sonora escalado** (dB) en función de parámetros físicos del flujo y la geometría.

### 🔬 Variables

**Predictoras:**
- frequency (Hz)  
- angle_of_attack (deg)  
- chord_length (m)  
- free_stream_velocity (m/s)  
- suction_side_displacement_thickness (m)

**Objetivo:**
- **sound_pressure_level** (dB)

### 📂 Archivo
```
data/raw/airfoil_self_noise.csv
```

---

# 🛠️ 3. Datos procesados

Los archivos generados tras preprocesado (escalado, normalización, imputación, generación de folds, etc.) se almacenan en:

```
data/processed/
```

Estos se generan mediante los scripts del directorio:

```
scripts/
```

---

# 🔄 4. Reproducibilidad

Para descargar los datos automáticamente:

```
python scripts/download_datasets.py
```

---

# 📜 5. Licencia y uso

Ambos datasets son públicos y provienen del **UCI Machine Learning Repository**.  
Se utilizan exclusivamente con fines académicos en el contexto del TFM.
