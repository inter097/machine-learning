import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Cargar modelos
modelos = {
    'Regresión Logística': joblib.load('modelos/modelo_log.pkl'),
    'Naive Bayes': joblib.load('modelos/modelo_nb.pkl'),
    'KNN': joblib.load('modelos/modelo_knn.pkl'),
    'Árbol de Decisión': joblib.load('modelos/modelo_arbol.pkl'),
    'SVM': joblib.load('modelos/modelo_svm.pkl'),
    'Red Neuronal MLP': joblib.load('modelos/modelo_mlp.pkl')  # 👈 nuevo modelo
}


# Cargar scaler único
scaler = joblib.load('modelos/scaler.pkl')

# Opcional: cargar datos de test (para mostrar matriz de confusión)
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

# Inputs reales del usuario
edad = st.number_input("Edad", min_value=0, max_value=120, value=45)
ckmb = st.number_input("CK-MB", value=2.86, min_value=0.00, format="%.2f")
troponina = st.number_input("Troponina", value=0.003, min_value=0.000, format="%.3f", step=0.001)

# Transformar datos 1e-10 es para evitar log(0) de infinito, 1e-10 es igual a 0.0000000001
ckmb_log = np.log(ckmb + 1e-10)
troponina_log = np.log(troponina + 1e-10)
entrada = np.array([[edad, ckmb_log, troponina_log]])
entrada_scaled = scaler.transform(entrada)

# Crear pestañas
tabs = st.tabs(list(modelos.keys()))

for idx, nombre_modelo in enumerate(modelos.keys()):
    with tabs[idx]:
        # st.subheader(nombre_modelo)
        modelo = modelos[nombre_modelo]

        # Predicción
        pred = modelo.predict(entrada_scaled)[0]
        resultado = "Positivo" if pred == 1 else "Negativo"
        st.markdown(f"### Resultado: **{resultado}**")

        # Probabilidad (si aplica)
        try:
            proba = modelo.predict_proba(entrada_scaled)[0]
            valor_1 = round(proba[0] * 100)
            valor_2 = round(proba[1] * 100)
            st.write("Probabilidades:")
            st.write(f"- Negativo: {valor_1}%")
            st.write(f"- Positivo: {valor_2}%")
        except:
            st.info("Este modelo no proporciona probabilidades (`predict_proba`).")

        # Mostrar matriz de confusión
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