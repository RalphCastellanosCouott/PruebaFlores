import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import json

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Flores",
    page_icon="🌸",
    layout="centered"
)

# Título de la aplicación
st.title("🌸 Clasificador de Flores")
st.markdown("---")

# Cargar el modelo y las clases
@st.cache_resource
def load_model_and_classes():
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('modelo_flores.keras')
    
    # Cargar los nombres de las clases
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    return model, class_names

# Función para cargar imagen desde URL
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error al cargar la imagen: {e}")
        return None

# Función para preprocesar la imagen
def preprocess_image(img):
    # Redimensionar a 180x180 como se usó en el entrenamiento
    img = img.resize((180, 180))
    # Convertir a array y normalizar
    img_array = np.array(img) / 255.0
    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para hacer la predicción
def predict_image(model, img_array, class_names):
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Crear diccionario de probabilidades
    probabilities = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, probabilities

# Cargar modelo y clases
with st.spinner("Cargando modelo..."):
    model, class_names = load_model_and_classes()

st.success("✅ Modelo cargado correctamente!")

# Mostrar las clases disponibles
st.markdown("### 🌼 Clases disponibles:")
cols = st.columns(5)
for i, name in enumerate(class_names):
    cols[i].markdown(f"- {name.capitalize()}")

st.markdown("---")

# Input para la URL de la imagen
st.markdown("### 📷 Ingresa la URL de la imagen:")
url = st.text_input("URL de la imagen", placeholder="https://ejemplo.com/flor.jpg")

# Botón para clasificar
if st.button("🔍 Clasificar flor", type="primary"):
    if url:
        with st.spinner("Cargando y analizando la imagen..."):
            # Cargar imagen desde URL
            img = load_image_from_url(url)
            
            if img is not None:
                # Crear dos columnas: una para la imagen y otra para los resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🖼️ Imagen cargada:")
                    st.image(img, use_container_width=True)
                
                # Preprocesar y predecir
                img_array = preprocess_image(img)
                
                # DEPURACIÓN: Ver los valores de la imagen procesada
                st.write(f"Forma de la imagen: {img_array.shape}")
                st.write(f"Rango de valores: min={img_array.min():.4f}, max={img_array.max():.4f}")
                st.write(f"Valor promedio: {img_array.mean():.4f}")

                # Mostrar las probabilidades de todas las clases
                predictions = model.predict(img_array, verbose=0)[0]
                for i, name in enumerate(class_names):
                    st.write(f"{name}: {predictions[i]:.4f}")

                predicted_class, confidence, probabilities = predict_image(model, img_array, class_names)
                
                with col2:
                    st.markdown("### 🎯 Resultado:")
                    # Resaltar la clase más probable
                    st.markdown(f"""
                    <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: white;">{predicted_class.upper()}</h2>
                        <p style="color: white; font-size: 18px;">Confianza: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar distribución de probabilidades
                st.markdown("---")
                st.markdown("### 📊 Distribución de probabilidades:")
                
                # Crear gráfico de barras con las probabilidades
                import pandas as pd
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 4))
                classes = list(probabilities.keys())
                probs = list(probabilities.values())
                
                bars = ax.bar(classes, probs, color='skyblue')
                # Resaltar la barra de la clase predicha
                predicted_idx = classes.index(predicted_class)
                bars[predicted_idx].set_color('#4CAF50')
                
                ax.set_ylabel('Probabilidad')
                ax.set_title('Probabilidades por clase')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Mostrar tabla de probabilidades
                st.markdown("### 📋 Detalle por clase:")
                prob_df = pd.DataFrame({
                    'Clase': classes,
                    'Probabilidad': [f"{p:.2%}" for p in probs]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
    else:
        st.warning("⚠️ Por favor, ingresa una URL de imagen válida.")

# Footer
st.markdown("---")
st.markdown("💡 **Nota:** El modelo fue entrenado con imágenes de daisy, dandelion, roses, sunflowers y tulips.")