# Resumen de avances (1 página)

**Objetivo del avance:** demostrar prueba de concepto del método propuesto comparado con baseline.

**Contenido entregado:**  
- Notebook con EDA y pipeline baseline (`notebooks/01_EDA.py` / `01_EDA_clean.ipynb`)  
- Ficha de datos (`data/README.md`)  
- Figuras principales en `figures/`  
- Instrucciones de reproducción en `README.md`

**Método:**  
- Preprocesado: eliminación duplicados, imputación por mediana en numéricas, codificación simple en categóricas.  
- Pipeline: baseline (Dummy) vs modelo propuesto (RandomForest como ejemplo).  
- Validación: partición train/test 80/20 y validación cruzada K‑fold.

**Resultados preliminares:**  
- Métrica baseline: [valor]  
- Métrica modelo: [valor]  
- Interpretación breve: el modelo muestra [mejora / no mejora] en la métrica X; se requiere ajuste de features y pruebas estadísticas para confirmar significación.

**Próximos pasos:**  
1. Refinar ingeniería de variables.  
2. Probar alternativas de modelos y ajuste de hiperparámetros.  
3. Realizar pruebas estadísticas sobre folds.  
4. Redactar metodología y resultados para el TFM.
