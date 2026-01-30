# -*- coding: utf-8 -*-
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models_ixe")

st.title("Calculadora predictiva Ixekizumab (semana 16)")
st.caption(
    "Herramienta de apoyo a la decisión clínica. "
    "Modelo entrenado con datos observacionales. "
    "No sustituye al juicio clínico."
)

with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

pasi = st.number_input("PASI basal", 0.0, 80.0, 20.0)
edad = st.number_input("Edad", 18, 100, 45)
imc = st.number_input("IMC", 15.0, 60.0, 27.0)
sexo = st.selectbox("Sexo", ["Varón", "Mujer"])
artritis = st.selectbox("Artritis psoriásica", ["No", "Sí"])
anios = st.number_input("Años con psoriasis", 0, 80, 15)
nprev = st.number_input("Nº biológicos previos", 0, 20, 0)

objetivo = st.selectbox("Objetivo", ["PASI75_w16", "PASI90_w16"])

if st.button("Calcular probabilidad"):
    # Cargar modelos
    model_75 = joblib.load(MODELS_DIR / "ixe_PASI75_w16.joblib")
    model_90 = joblib.load(MODELS_DIR / "ixe_PASI90_w16.joblib")

    # Construir input EXACTO
    X = pd.DataFrame([{
        "PASI INICIAL IXE": float(pasi),
        "edad": int(edad),
        "IMC": float(imc),
        "Sexo": sexo,
        "ARTRITIS PSORIASICA": 1 if artritis == "Sí" else 0,
        "años con psoriasis": float(anios),
        "N biológicos previos": int(nprev),
    }])

    # Alinear columnas a cada modelo
    X75 = X.reindex(columns=model_75.feature_names_in_)
    X90 = X.reindex(columns=model_90.feature_names_in_)

    # Predicciones
    prob75 = model_75.predict_proba(X75)[0, 1]
    prob90 = model_90.predict_proba(X90)[0, 1]

    # Mostrar resultados
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilidad PASI75 (semana 16)",
            f"{prob75*100:.1f}%"
        )

    with col2:
        st.metric(
            "Probabilidad PASI90 (semana 16)",
            f"{prob90*100:.1f}%"
        )

    # Interpretación clínica simple (opcional pero recomendable)
    if prob90 >= 0.7:
        st.success("Alta probabilidad de respuesta profunda (PASI90).")
    elif prob90 >= 0.4:
        st.warning("Probabilidad intermedia de respuesta profunda (PASI90).")
    else:
        st.error("Baja probabilidad de alcanzar PASI90.")


st.markdown("---")

with st.expander("Transparencia del modelo"):
    st.write(
        "Modelo entrenado con datos observacionales de práctica clínica real. "
        "Se muestran variables utilizadas y métricas internas del modelo."
    )

    metadata_path = MODELS_DIR / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        st.subheader("Variables incluidas")
        st.write(", ".join(meta.get("features", [])))

        st.subheader("Rendimiento interno")
        models_info = meta.get("models", {})

        for endpoint, info in models_info.items():
            st.markdown(f"**{endpoint}**")

            # Metadata antiguo → solo ruta
            if isinstance(info, str):
                st.caption(f"Archivo del modelo: {info}")
                st.info(
                    "Este modelo no incluye métricas internas (AUC, Brier). "
                    "Para mostrarlas es necesario reentrenar el modelo."
                )
                continue

            # Metadata nuevo → métricas
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N", info.get("n", "NA"))
            c2.metric("Eventos", info.get("pos", "NA"))

            auc = info.get("auc")
            brier = info.get("brier")

            c3.metric("AUC", f"{auc:.2f}" if isinstance(auc, (int, float)) else "NA")
            c4.metric("Brier", f"{brier:.3f}" if isinstance(brier, (int, float)) else "NA")

        st.caption("AUC: discriminación. Brier: calibración (menor es mejor).")

    else:
        st.warning(
            "No se encontró el archivo metadata.json del modelo. "
            "Reentrena y vuelve a subir los modelos."
        )

    st.caption(
        "Herramienta de apoyo a la decisión clínica. "
        "No sustituye el juicio clínico."
    )
