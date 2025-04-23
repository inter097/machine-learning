import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

# Cargar modelos
modelos = {
    'Regresión Logística': joblib.load('modelos/modelo_log.pkl'),
    'Naive Bayes': joblib.load('modelos/modelo_nb.pkl'),
    'KNN': joblib.load('modelos/modelo_knn.pkl'),
    'Árbol de Decisión': joblib.load('modelos/modelo_arbol.pkl'),
    'SVM': joblib.load('modelos/modelo_svm.pkl'),
    'Red Neuronal MLP': joblib.load('modelos/modelo_mlp.pkl')
}

# Cargar scaler
scaler = joblib.load('modelos/scaler.pkl')

# Cargar datos de test
try:
    X_test = joblib.load('modelos/X_test_scaled.pkl')  # escalado
    y_test = joblib.load('modelos/y_test.pkl')
    mostrar_matriz = True
except:
    mostrar_matriz = False

# Interfaz
st.title("Predicción de Ataques Cardíacos")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
Esta aplicación utiliza técnicas de **Aprendizaje Automático (Machine Learning)** para **predecir el riesgo de un ataque cardíaco** en pacientes, a partir de tres biomarcadores clínicos fundamentales:

- **Edad:** Factor de riesgo asociado al desgaste cardiovascular.
- **CK-MB:** Enzima liberada ante daño en el músculo cardíaco.
- **Troponina:** Proteína clave cuya presencia elevada indica daño al corazón.

---

### Funciones principales:

- **Predicción** -> Realiza la predicción de reisgo de ataque cardíaco para el paciente evaluado.
- **Pobabilidad** -> muestra la probabilidad en un gráfico tipo pastel.
- **Matriz de confusión** -> Visualiza la para evaluar el desempeño del modelo.
- **Curvas ROC** -> Grafica que muestra la relación entre la tasa de TP y FP de todos los modelos.
- **Tiempo de inferencia** -> Evalua el **tiempo de inferencia** para cada modelo.
            
---
            
### Modelos implementados:
  - Regresión Logística
  - Naive Bayes
  - KNN
  - Árbol de Decisión
  - SVM
  - Red Neuronal MLP
---
            
### Metricas de evaluación:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **F0.5-score**
  - **F2-score**

---

### Uso de la aplicación

""")

if not mostrar_matriz:
    st.markdown("**Nota:** Las matrices de confusión no están disponibles porque no se cargaron los datos de test.")

# Inputs del usuario
edad = st.number_input("**Edad**", min_value=0, max_value=120, value=45)
ckmb = st.number_input("**CK-MB**", value=2.86, min_value=0.00, format="%.2f")
troponina = st.number_input("**Troponina**", value=0.003, min_value=0.000, format="%.3f", step=0.001)

st.markdown("---")
st.subheader("Eliga el modelo a utilizar")

# Transformar datos
ckmb_log = np.log(ckmb + 1e-10)
troponina_log = np.log(troponina + 1e-10)
entrada = np.array([[edad, ckmb_log, troponina_log]])
entrada_scaled = scaler.transform(entrada)
# Pestañas por modelo
tabs = st.tabs(list(modelos.keys()))

for idx, nombre_modelo in enumerate(modelos.keys()):
    with tabs[idx]:
        modelo = modelos[nombre_modelo]
        pred = modelo.predict(entrada_scaled)[0]

        st.markdown("")
        resultado = "Positivo" if pred == 1 else "Negativo"
        st.markdown(f"### {nombre_modelo} &nbsp;&nbsp;|&nbsp;&nbsp; Resultado → **{resultado}**")


        try:
            proba = modelo.predict_proba(entrada_scaled)[0]
            proba = [round(p, 2) for p in proba] # redondear a 2 decimales
        except:
            st.info("Este modelo no proporciona probabilidades (`predict_proba`).")

        if mostrar_matriz:
            y_pred = modelo.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

####################################################################
# muestra la matriz de confursión
            with st.container():
                # Fila completa (una línea visual)
                col1, col2, col3 = st.columns([1, 3,1])  # Ajusta proporción según gusto

               


####################################################################
# grafico de pastel con la probabilidad de predicción

                with col2:
                    try:
                        fig_plotly = go.Figure(data=[go.Pie(
                            labels=['Positivo','Negativo'],
                            values=[proba[1], proba[0]],
                            marker=dict(colors=['c62828' ,'e0f7fa']),
                            sort=False
                        )])
                        fig_plotly.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=True
                        )
                        st.plotly_chart(fig_plotly, use_container_width=True, key=f"plotly_{nombre_modelo}")
                    except Exception as e:
                        st.error(f"Error al generar el gráfico: {e}")


st.markdown("---")


st.subheader("Opciones de visualización")

opcion_vista = st.radio(
    "",
    [
        "Resumen de métricas",
        "Curvas ROC",
        "Matrices de confusión",
        "Tiempo de inferencia"
    ],horizontal=True,
    index=0
)


######################################################################
# Resumen de métricas

if mostrar_matriz:
    resumen = []
    tiempos = {}
    matrices = {}
    roc_data = {}

    for nombre, modelo in modelos.items():
        try:
            start = time.perf_counter()
            y_pred = modelo.predict(X_test)
            tiempos[nombre] = time.perf_counter() - start

            # Para curva ROC
            if hasattr(modelo, "predict_proba"):
                y_score = modelo.predict_proba(X_test)[:, 1]
            elif hasattr(modelo, "decision_function"):
                y_score = modelo.decision_function(X_test)
            else:
                y_score = None

            if y_score is not None:
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                roc_data[nombre] = (fpr, tpr, roc_auc)

            # Calcular métricas extendidas
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            f1 = f1_score(y_test, y_pred)
            f2 = (5 * prec * rec) / ((4 * prec) + rec) if (4 * prec + rec) != 0 else 0
            f05 = (1.25 * prec * rec) / ((0.25 * prec) + rec) if (0.25 * prec + rec) != 0 else 0

            resumen.append({
                "Modelo": nombre,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1,
                "F0.5-score": f05,
                "F2-score": f2
            })

            matrices[nombre] = confusion_matrix(y_test, y_pred)
        except:
            continue

    # Ordenar resumen por Accuracy descendente
    resumen_ordenado = sorted(resumen, key=lambda x: x["Accuracy"], reverse=True)

    # guardar el orden de los modelos
    orden_modelos = [r["Modelo"] for r in resumen_ordenado]

    # reordenar los diccionarios de tiempos, matrices y roc_data
    matrices_ordenadas = {nombre: matrices[nombre] for nombre in orden_modelos}

    df_resumen_ordenado = pd.DataFrame(resumen_ordenado).set_index("Modelo")

    if opcion_vista == "Resumen de métricas":
        st.markdown("**Escala de color:** rendimiento bajo (claro) → rendimiento alto (oscuro)")
        st.dataframe(
            df_resumen_ordenado.style
                .format("{:.2%}")
                .background_gradient(cmap='Blues', axis=0)
        )

        st.markdown("-----------------------")

        st.markdown("**Tabla de Métricas de Evaluación**")

        st.markdown("""
        | **Métrica**   | **Descripción**                                                       | **Fórmula**                                               |
        |---------------|------------------------------------------------------------------------|------------------------------------------------------------|
        | Accuracy      | Porcentaje total de predicciones correctas.                          | $\\frac{TP + TN}{TP + TN + FP + FN}$                      |
        | Precision     | Proporción de positivos predichos que son correctos.                 | $\\frac{TP}{TP + FP}$                                     |
        | Recall        | Proporción de positivos reales correctamente identificados.          | $\\frac{TP}{TP + FN}$                                     |
        | F1-score      | Promedio armónico equilibrado entre precisión y recall.              | $2\\cdot \\frac{\\text{P} \\cdot \\text{R}}{\\text{P} + \\text{R}}$|
        | F0.5-score    | Promedio armónico con mayor peso en la precisión.                    | $1.25\\cdot \\frac{\\text{P} \\cdot \\text{R}}{0.25 \\cdot \\text{P} + \\text{R}}$ |
        | F2-score      | Promedio armónico con mayor peso en el recall.                       | $5\\cdot \\frac{\\text{P} \\cdot \\text{R}}{4 \\cdot \\text{P} + \\text{R}}$ |
        """, unsafe_allow_html=False)



######################################################################
    # Curvas ROC
    elif opcion_vista == "Curvas ROC":
        fig, ax = plt.subplots()

        roc_data_ordenado = sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True)
        
        for nombre, (fpr, tpr, roc_auc) in roc_data_ordenado:
            ax.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', label="Random")
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        ax.set_title('Curvas ROC')
        ax.legend(loc='lower right')
        st.pyplot(fig)

        st.markdown("Las curvas ROC muestran la relación entre la tasa de verdaderos positivos y la tasa de falsos positivos.")
        st.markdown("**AUC:** área bajo la curva, indica el rendimiento del modelo. Un AUC de 1.0 es perfecto, 0.5 es aleatorio.")

######################################################################
    # Matrices de confusión

    elif opcion_vista == "Matrices de confusión":   
        st.markdown("Las matrices de confusión muestran la relación entre las predicciones y los valores reales.")
        # Mostrar 3 por matrices de confusión en una fila
        
        modelos_lista = list(matrices_ordenadas.items())

        for i in range(0, len(modelos_lista), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(modelos_lista):
                    nombre, cm = modelos_lista[i + j]
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4, 4))  # Tamaño ajustado
                        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                                    xticklabels=["Negativo", "Positivo"],
                                    yticklabels=["Negativo", "Positivo"], ax=ax, annot_kws={"size": 14})
                        ax.set_title(f"{nombre}", fontsize=10)
                        ax.tick_params(labelsize=8)
                        # Etiquetas manuales tipo marca de agua
                        labels = [['TN', 'FP'], ['FN', 'TP']]
                        for k in range(2):
                            for l in range(2):
                                ax.text(l + 0.5, k + 0.5, labels[k][l],
                                        color='gray', fontsize=44, ha='center', va='center', alpha=0.3)
                        st.pyplot(fig)

######################################################################
    # Tiempo de inferencia

    elif opcion_vista == "Tiempo de inferencia":
        import plotly.express as px
        import pandas as pd

         # Ordenar por  el menor tiempo de inferencia 
        tiempos_ordenados = sorted(tiempos.items(), key=lambda x: x[1])

        # Convertir el diccionario a DataFrame
        df_tiempos = pd.DataFrame(tiempos_ordenados,columns=['Modelo', 'Tiempo'])

        # Crear gráfico interactivo
        fig_plotly = px.bar(
            df_tiempos,
            x='Modelo',
            y='Tiempo',
            color='Tiempo',
            color_continuous_scale='Blues',
            text=df_tiempos['Tiempo'].apply(lambda x: f"{x:.4f}s"),
            title="Tiempo de predicción por modelo",
            labels={'Tiempo': 'Segundos'}
        )

        fig_plotly.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Segundos",
            coloraxis_showscale=False,
            margin=dict(t=100)  # aumenta espacio superior
        )
        fig_plotly.update_traces(textposition='outside', cliponaxis=False)


        # Mostrar en Streamlit
        st.plotly_chart(fig_plotly, use_container_width=True)

    
st.markdown("")
st.markdown("")
st.markdown("""
<hr style="margin-top: 30px; margin-bottom:60px;">

<h4 style='text-align: left;'>Conclusión:</h4>

<div style='text-align: justify; margin-bottom: 200px;'>
Esta aplicación es una herramienta interactiva para la <b>predicción de ataques cardíacos</b>. 
Permite evaluar el riesgo cardiovascular de los pacientes de manera rápida y precisa, facilitando la toma de decisiones informadas en el ámbito clínico. 
Es especialmente adecuada para entornos de apoyo clínico, educación en ciencia de datos o exploración de modelos predictivos.
</div>
""", unsafe_allow_html=True)


  


