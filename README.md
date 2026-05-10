# 📘 TFM — Regresión por Mínimos Cuadrados Parciales (PLS)  
### *Tratamiento computacional y comparación con métodos clásicos*

Este repositorio contiene el código completo y reproducible del Trabajo Fin de Máster del **Máster en Estadística Aplicada (UGR)**. El proyecto implementa un pipeline robusto para:

- Ajustar modelos de **Partial Least Squares Regression (PLS)**.  
- Seleccionar el número óptimo de componentes mediante **validación cruzada real**.  
- Comparar PLS con métodos lineales clásicos: **OLS, Ridge, Lasso y PCR**.  
- Ejecutar el análisis sobre tres datasets representativos:  
  - **Tecator** (quimiometría, espectros NIR)  
  - **Gasoline** (octanaje, espectros NIR)  
  - **Riboflavin** (alta dimensionalidad p≫n, genómica)

El proyecto está empaquetado como un **wheel instalable**, lo que permite reproducir todos los resultados con un único comando.

---

## 🐍 Requisitos

- **Python 3.10, 3.11 o 3.12**  
  (recomendado: **Python 3.10** por compatibilidad con librerías científicas)

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/repo_TFM.git
cd repo_TFM
```

### 2. (Opcional) Crear un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar el paquete

Desde el wheel:

```bash
pip install dist/repo_tfm-0.1.0-py3-none-any.whl
```

O directamente desde el repositorio:

```bash
pip install .
```

---

## ▶️ Ejecución del pipeline completo

Una vez instalado, basta con ejecutar:

```bash
python -m repo_tfm
```

Esto lanzará automáticamente:

- carga de los datasets  
- validación cruzada para PLS  
- entrenamiento del modelo final  
- ejecución de modelos clásicos  
- comparación de resultados  
- generación de métricas y gráficos  

Los resultados se guardarán en:

```
results/<dataset>/
    ├── csv/
    └── figures/
```

---

## 📁 Estructura del proyecto

```
repo_tfm/
├── data/
│   └── raw/                        # Datasets originales
│
├── src/repo_tfm/
│   ├── main.py                     # Pipeline principal
│   ├── scripts/
│   │   ├── core/                   # Utilidades internas (EDA, métricas, logger, etc.)
│   │   ├── models/                 # Modelos implementados
│   │   │   ├── ols_model.py
│   │   │   ├── ridge_model.py
│   │   │   ├── lasso_model.py
│   │   │   ├── pcr_model.py
│   │   │   └── pls_model.py
│   │   ├── experiments/            # Ejecución de experimentos
│   │   │   ├── classical_experiment.py
│   │   │   ├── pls_experiment.py
│   │   │   └── pls_final_experiment.py
│   │   └── compare_runner.py       # Orquestador de comparaciones
│
├── results/                        # Resultados generados automáticamente
├── dist/                           # Wheel y distribución
└── README.md
```

---

## 📊 Datasets incluidos

| Dataset | n | p | Dominio | Objetivo |
|--------|---|---|----------|----------|
| **Tecator** | 215 | 100 | Quimiometría | Predicción de grasa (`fat`) |
| **Gasoline** | 60 | 401 | Quimiometría | Predicción de octanaje |
| **Riboflavin** | 71 | 4088 | Genómica | Producción de riboflavina |

---

## 🤖 Modelos implementados

| Modelo | Descripción |
|--------|-------------|
| **PLS_CV** | Selección óptima de componentes mediante validación cruzada |
| **PLS_FINAL** | Entrenamiento final con el número óptimo de componentes |
| **PCR** | PCA + regresión |
| **OLS** | Regresión lineal clásica |
| **Ridge** | Regularización L2 |
| **Lasso** | Regularización L1 |

---

## 📈 Resultados esperados (resumen)

| Dataset | Mejor modelo | Interpretación |
|---------|--------------|----------------|
| **Gasoline** | OLS | Estructura lineal fuerte. |
| **Tecator** | PLS (≈5 comp.) | Caso clásico donde PLS domina. |
| **Riboflavin** | Lasso | p≫n extremo; sparsity es clave. |

---

## 🧪 Cómo extender el proyecto

### Añadir un nuevo dataset
1. Colocar el archivo en `data/raw/`.  
2. Añadir un método de carga en `data_loader.py`.  
3. Registrar el dataset en `main.py`.

### Añadir un nuevo modelo
1. Crear un archivo en `scripts/models/`.  
2. Implementar la clase del modelo.  
3. Integrarlo en los experimentos y en `compare_runner.py`.

---

## 📝 Licencia

Proyecto distribuido con fines académicos para garantizar reproducibilidad.

---

## 🙌 Contacto

**Autor:** Andrés  
**Máster:** Estadística Aplicada (UGR)  
**Año:** 2025–2026  
