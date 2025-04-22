import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
st.markdown("Este sistema predice la probabilidad de un ataque cardíaco en base a los valores de Edad, CK-MB y Troponina.")
st.markdown("Ingrese los valores **sin transformar**, el sistema los transformará internamente.")

# Inputs del usuario
edad = st.number_input("Edad", min_value=0, max_value=120, value=45)
ckmb = st.number_input("CK-MB", value=2.86, min_value=0.00, format="%.2f")
troponina = st.number_input("Troponina", value=0.003, min_value=0.000, format="%.3f", step=0.001)

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
        resultado = "Positivo" if pred == 1 else "Negativo"
        st.markdown(f"### Resultado: **{resultado}**")

        try:
            proba = modelo.predict_proba(entrada_scaled)[0]
            st.write("Probabilidades:")
            st.write(f"- Negativo: {round(proba[0]*100)}%")
            st.write(f"- Positivo: {round(proba[1]*100)}%")
        except:
            st.info("Este modelo no proporciona probabilidades (`predict_proba`).")

        if mostrar_matriz:
            y_pred = modelo.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                        xticklabels=["Negativo", "Positivo"],
                        yticklabels=["Negativo", "Positivo"], ax=ax)
            ax.set_title("Matriz de Confusión (test set)")
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
            st.pyplot(fig)

# ------------------ Comparación entre modelos ------------------

st.markdown("---")
st.subheader("🔍 Comparación entre modelos")

opcion_vista = st.radio(
    "Selecciona una visualización",
    [
        "Resumen de métricas",
        "Accuracy por modelo",
        "Matrices de confusión",
        "Tiempo de inferencia"
    ],
    index=0  # muestra por defecto "Resumen de métricas"
)

if mostrar_matriz:
    resumen = []
    tiempos = {}
    matrices = {}

    for nombre, modelo in modelos.items():
        try:
            start = time.time()
            y_pred = modelo.predict(X_test)
            tiempos[nombre] = time.time() - start
            resumen.append({
                "Modelo": nombre,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-score": f1_score(y_test, y_pred)
            })
            matrices[nombre] = confusion_matrix(y_test, y_pred)
        except:
            continue

    df_resumen = pd.DataFrame(resumen).set_index("Modelo")

    if opcion_vista == "Resumen de métricas":
        st.dataframe(df_resumen.style.format("{:.2%}"))

    elif opcion_vista == "Accuracy por modelo":
        fig, ax = plt.subplots()
        df_resumen["Accuracy"].plot(kind='bar', ax=ax)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy por modelo")
        st.pyplot(fig)

    elif opcion_vista == "Matrices de confusión":
        st.markdown("Visualización de las matrices de confusión por modelo.")
        cols = st.columns(len(matrices))
        for i, (nombre, cm) in enumerate(matrices.items()):
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                        xticklabels=["Negativo", "Positivo"],
                        yticklabels=["Negativo", "Positivo"], ax=ax)
            ax.set_title(nombre)
            cols[i].pyplot(fig)

    elif opcion_vista == "Tiempo de inferencia":
        fig, ax = plt.subplots()
        pd.Series(tiempos).plot(kind='bar', ax=ax)
        ax.set_ylabel("Segundos")
        ax.set_title("Tiempo de predicción por modelo")
        st.pyplot(fig)
