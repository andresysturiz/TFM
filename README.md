# 📘 TFM — Regresión por Mínimos Cuadrados Parciales (PLS)
### *Tratamiento computacional y comparación con métodos clásicos*

Este repositorio contiene el código completo del Trabajo Fin de Máster del **Máster en Estadística Aplicada (UGR)**, centrado en:

- Implementación rigurosa de **Partial Least Squares Regression (PLS)**.  
- Selección óptima de componentes mediante **validación cruzada real**.  
- Comparación con modelos clásicos: **OLS, Ridge, Lasso, PCR**.  
- Aplicación a tres datasets representativos:  
  - **Tecator** (quimiometría, espectros NIR)  
  - **Gasoline** (octanaje, espectros NIR)  
  - **Riboflavin** (alta dimensionalidad p≫n, genómica)

El proyecto está empaquetado como un **wheel instalable**, permitiendo reproducir todos los resultados con un solo comando.

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/repo_TFM.git
cd repo_TFM
```

### 2. Crear entorno virtual (opcional)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar el paquete desde el wheel

```bash
pip install dist/repo_tfm-0.1.0-py3-none-any.whl
```

O instalarlo directamente desde el repositorio:

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
- validación cruzada PLS (PLS_CV)  
- entrenamiento del modelo final PLS (PLS_FINAL)  
- ejecución de modelos clásicos  
- comparación final  
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
repo_TFM/
├── data/
│   ├── raw/                # Datasets originales
│   └── processed/          # (opcional) datos transformados
│
├── dist/                   # Wheel generado
│   ├── repo_tfm-0.1.0-py3-none-any.whl
│   └── repo_tfm-0.1.0.tar.gz
│
├── results/                # Resultados generados automáticamente
│   ├── gasoline/
│   ├── riboflavin/
│   └── tecator/
│
├── src/repo_tfm/
│   ├── main.py             # Pipeline principal
│   └── scripts/
│       ├── core/           # Utilidades internas
│       └── experiments/    # Experimentos PLS y modelos clásicos
│
├── notebooks/              # Análisis visual opcional
├── references/             # Bibliografía
├── docs/                   # Documentación adicional
└── README.md
```

---

## 📊 Datasets incluidos

| Dataset | n | p | Dominio | Objetivo |
|--------|---|---|----------|----------|
| **Tecator** | 215 | 100 | Quimiometría | Predicción de grasa (`fat`) |
| **Gasoline** | 60 | 401 | Quimiometría | Predicción de octanaje |
| **Riboflavin** | 71 | 4088 | Genómica (p≫n) | Producción de riboflavina (`x`) |

---

## 🤖 Modelos implementados

| Modelo | Descripción |
|--------|-------------|
| **PLS_CV** | Selección óptima de componentes mediante validación cruzada real |
| **PLS_FINAL** | Entrenamiento final con el número óptimo de componentes |
| **PCR** | PCA + regresión |
| **OLS** | Regresión lineal clásica |
| **Ridge** | Regularización L2 |
| **Lasso** | Regularización L1 |

---

## 📈 Resultados esperados (resumen)

| Dataset | Mejor modelo | Interpretación |
|---------|--------------|----------------|
| **Gasoline** | OLS | Estructura lineal fuerte; OLS domina. |
| **Tecator** | PLS (5 comp.) | Dataset clásico donde PLS es superior. |
| **Riboflavin** | Lasso | p≫n extremo; sparsity domina. |

---

## 🧪 Cómo extender el proyecto

### Añadir un nuevo dataset
1. Colocar el archivo en `data/raw/`.
2. Añadir un método en `data_loader.py`.
3. Añadir el nombre en `main.py`.

### Añadir un nuevo modelo
1. Crear un archivo en `scripts/experiments/`.
2. Implementar `.run()`.
3. Añadirlo en `compare_runner.py`.

---

## 📚 Referencias clave

- Geladi & Kowalski (1986). *Partial Least Squares: A Tutorial*.  
- Frank & Friedman (1993). *A Statistical View of Chemometrics Tools*.  
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*.  

---

## 📝 Licencia

Este proyecto se distribuye con fines académicos para reproducibilidad del TFM.

---

## 🙌 Contacto

**Autor:** Andrés  
**Máster:** Estadística Aplicada (UGR)  
**Año:** 2025–2026  
