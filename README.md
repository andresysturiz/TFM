# TFM
Ejecucion de modelos lineales clasicos vs al pls
=======
# 📘 Trabajo Fin de Máster  
## Comparación de Partial Least Squares (PLS) con métodos clásicos de regresión  
### Aplicación a los datasets *Wine Quality* y *Airfoil Self‑Noise*

Este repositorio contiene todo el código, datos, documentación y resultados generados durante el desarrollo del Trabajo Fin de Máster del Máster en Estadística Aplicada.  
El objetivo principal es comparar el rendimiento e interpretabilidad de **Partial Least Squares (PLS)** frente a métodos clásicos de regresión en dos dominios distintos: química y aeroacústica.

---

# 📁 Estructura del repositorio

```
repo_TFM/
 ├── data/
 │   ├── raw/          # Datos originales descargados desde UCI
 │   └── processed/    # Datos preprocesados (escalados, folds, etc.)
 ├── scripts/          # Scripts ejecutables del pipeline
 ├── notebooks/        # Notebooks de análisis y visualización
 ├── figures/          # Gráficos generados para el TFM
 ├── docs/             # Documentación adicional
 ├── references/       # Artículos y bibliografía
 ├── pyproject.toml    # Gestión de dependencias (Poetry)
 ├── poetry.lock
 └── README.md         # Este archivo
```

---

# 🎯 Objetivo del proyecto

Evaluar el comportamiento de **PLS** frente a métodos clásicos de regresresión:

- **OLS (Regresión lineal)**
- **Ridge**
- **Lasso**
- **PCR (Principal Component Regression)**
- **Random Forest Regressor** (como referencia no lineal)

Se analizan dos datasets con naturaleza física distinta:

1. **Wine Quality** → dominio químico  
2. **Airfoil Self‑Noise** → dominio aeroacústico  

Esto permite estudiar la robustez, interpretabilidad y estabilidad de PLS en contextos variados.

---

# 📊 Datasets

Los datasets utilizados provienen del **UCI Machine Learning Repository** y se documentan en detalle en:

```
data/README.md
```

Los datos se descargan automáticamente mediante:

```
python scripts/download_datasets.py
```

---

# 🛠️ Pipeline del proyecto

El flujo de trabajo completo se organiza mediante scripts en `scripts/`:

- `download_datasets.py` → descarga y guarda los datos en `data/raw/`
- `preprocess.py` → limpieza, escalado y generación de datos procesados
- `export_folds.py` → creación de folds reproducibles para CV
- `pls_cv.py` → ajuste de modelos PLS con validación cruzada
- `compare_methods.py` → comparación con métodos clásicos
- `utils/` (si aplica) → funciones auxiliares

Los análisis exploratorios, visualizaciones y resultados interpretables se encuentran en:

```
notebooks/
```

---

# 📦 Dependencias

El proyecto utiliza **Poetry** para la gestión de entornos y dependencias.

Para instalar el entorno:

```
poetry install
```

Para activar el entorno virtual:

```
poetry shell
```

---

# 📈 Resultados

Los resultados (tablas, métricas, gráficos) se almacenan en:

```
figures/
results/   (si se añade)
```

Incluyen:

- Comparación de MSE, RMSE, R² entre modelos  
- Selección óptima de componentes en PLS  
- Loadings, scores y análisis interpretativo  
- Comparación entre dominios (química vs aeroacústica)

---

# 📚 Referencias

Las referencias bibliográficas utilizadas en el TFM se encuentran en:

```
references/
```

Incluyen artículos sobre:

- Partial Least Squares  
- PCR  
- Comparaciones metodológicas  
- Aplicaciones en química y aeroacústica  

---

# 🔄 Reproducibilidad

Para reproducir el proyecto desde cero:

1. Clonar el repositorio  
2. Instalar dependencias con Poetry  
3. Descargar los datos  
4. Ejecutar el pipeline o abrir los notebooks  

```
poetry install
poetry shell
python scripts/download_datasets.py
```

---

# 📝 Licencia

Los datasets utilizados son públicos y provienen del **UCI Machine Learning Repository**.  
El código del proyecto se utiliza exclusivamente con fines académicos.

```
>>>>>>> a7db1b3 (Añadida carpeta R y scripts iniciales)
