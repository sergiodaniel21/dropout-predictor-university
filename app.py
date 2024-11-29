# Importar las librerías necesarias
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo previamente entrenado y el LabelEncoder
model = joblib.load('modelo_clasificacion.pkl')  # Cargar el modelo Random Forest
label_encoder = joblib.load('label_encoder.pkl')  # Cargar el LabelEncoder que se usó en el entrenamiento

# Cargar el dataset original para obtener las categorías únicas
dataset = pd.read_csv('data.csv', sep=',', encoding='latin1')  # Asegúrate de que el dataset esté en la ruta correcta

# Título de la aplicación Streamlit
st.title("Predicción de Categoría Objetivo")

# Mostrar una vista previa del dataset
if st.checkbox("Ver los primeros registros del dataset"):
    st.write(dataset.head())

# Crear formulario para ingresar los datos de entrada
st.write("Introduce los siguientes datos para hacer una predicción:")

# Entradas del formulario para las características
estado_civil = st.selectbox('Estado Civil', dataset['ESTADO_CIVIL'].unique(), key='estado_civil_selectbox')
colegio_depa = st.selectbox('Colegio de Departamento', dataset['COLEGIO_DEPA'].unique(), key='colegio_depa_selectbox')
grado_instruccion_madre = st.selectbox('Grado de Instrucción de la Madre', dataset['GRADO_INSTRUCCION_MADRE'].unique(), key='grado_instruccion_madre_selectbox')
grado_instruccion_padre = st.selectbox('Grado de Instrucción del Padre', dataset['GRADO_INSTRUCCION_PADRE'].unique(), key='grado_instruccion_padre_selectbox')
necesidades_educativas = st.selectbox('Necesidades Educativas Especiales', dataset['NECESIDADES_EDUCATIVAS_ESPECIALES'].unique(), key='necesidades_educativas_selectbox')
nota_promedio = st.number_input('Nota Promedio', min_value=0.0, max_value=20.0, value=15.0, key='nota_promedio_numberinput')
domicilio_depa = st.selectbox('Domicilio Departamento', dataset['DOMICILIO_DEPA'].unique(), key='domicilio_depa_selectbox')
anio_nacimiento = st.number_input('Año de Nacimiento', min_value=1900, max_value=2024, value=1995, key='anio_nacimiento_numberinput')
nacimiento_depa = st.selectbox('Nacimiento Departamento', dataset['NACIMIENTO_DEPA'].unique(), key='nacimiento_depa_selectbox')
sexo = st.selectbox('Sexo', dataset['SEXO'].unique(), key='sexo_selectbox')  # Asumir que 1 es masculino y 2 es femenino
modalidad = st.selectbox('Modalidad', dataset['MODALIDAD'].unique(), key='modalidad_selectbox')
facultad = st.selectbox('Facultad', dataset['FACULTAD'].unique(), key='facultad_selectbox')
especialidad = st.selectbox('Especialidad', dataset['ESPECIALIDAD'].unique(), key='especialidad_selectbox')
ciclo_relativo = st.number_input('Ciclo Relativo', min_value=1, max_value=10, value=1, key='ciclo_relativo_numberinput')

# Crear un botón para realizar la predicción
if st.button('Predecir'):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame([[estado_civil, colegio_depa, grado_instruccion_madre, grado_instruccion_padre,
                                necesidades_educativas, nota_promedio, domicilio_depa, anio_nacimiento,
                                nacimiento_depa, sexo, modalidad, facultad, especialidad, ciclo_relativo]],
                              columns=['ESTADO_CIVIL', 'COLEGIO_DEPA', 'GRADO_INSTRUCCION_MADRE', 'GRADO_INSTRUCCION_PADRE', 
                                        'NECESIDADES_EDUCATIVAS_ESPECIALES', 'NOTA_PROMEDIO', 'DOMICILIO_DEPA', 'ANIO_NACIMIENTO', 
                                        'NACIMIENTO_DEPA', 'SEXO', 'MODALIDAD', 'FACULTAD', 'ESPECIALIDAD', 'CICLO_RELATIVO'])

    # Preprocesar las columnas categóricas usando el mismo LabelEncoder
    for col in input_data.columns:
        if input_data[col].dtype == 'object':  # Si la columna es categórica
            # Transformar las columnas usando el LabelEncoder cargado
            input_data[col] = label_encoder.transform(input_data[col])

    # Realizar la predicción usando el modelo cargado
    prediction = model.predict(input_data)

    # Mostrar la predicción
    st.write(f"La predicción para la categoría objetivo es: {prediction[0]}")

    # Mostrar una recomendación si la predicción es "DESERTADO"
    if prediction[0] == "DESERTADO":
        st.write("""
        **Recomendación:**
        
        Lamentablemente, los datos indican que el alumno ha sido clasificado como "DESERTADO". Esto podría estar relacionado con un bajo rendimiento académico o dificultades personales. 
        Se recomienda que el alumno reciba orientación académica y/o emocional, además de explorar opciones de apoyo institucional como tutorías, programas de ayuda psicológica o de acompañamiento académico para mejorar su situación y evitar que la deserción se haga efectiva.
        """)
