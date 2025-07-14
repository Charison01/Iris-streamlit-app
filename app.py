import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# Load Iris metadata for label mapping
iris = load_iris()

# Dictionary mapping species names to image URLs
species_images = {
    'setosa': 'https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg',
    'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
}

# Cache model and scaler loading for performance
@st.cache_resource
def load_artifacts(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts('best_mlp_model.pkl', 'scaler.pkl')

# Initialize session state for features
default_vals = {'Sepal Length': 5.1, 'Sepal Width': 3.5, 'Petal Length': 1.4, 'Petal Width': 0.2}
if 'features' not in st.session_state:
    st.session_state.features = default_vals.copy()

# Callback to sync slider changes to session_state.features
def update_feature(feature_name):
    st.session_state.features[feature_name] = st.session_state[feature_name]

st.set_page_config(page_title='Iris Classifier', layout='centered')
st.title('🌸 Iris Flower Species Classification')
st.write('Adjust the inputs or edit the table to predict species.')

# Side-by-side layout for inputs and prediction
input_col, result_col = st.columns([2, 1])

with input_col:
    sliders_col, table_col = st.columns(2)

    # Sliders
    with sliders_col:
        st.subheader('Adjust via Sliders')
        for feat, val in st.session_state.features.items():
            st.slider(
                label=f'{feat} (cm)',
                min_value=0.0,
                max_value=10.0,
                value=val,
                step=0.1,
                key=feat,
                on_change=update_feature,
                args=(feat,)
            )

    # Transposed editable table
    with table_col:
        st.subheader('Edit Directly')
        df = pd.DataFrame.from_dict(st.session_state.features, orient='index', columns=['Value'])
        df.index.name = 'Feature'
        edited = st.data_editor(df, num_rows='fixed', use_container_width=True, row_height=90)
        for feat in st.session_state.features:
            st.session_state.features[feat] = float(edited.at[feat, 'Value'])

# Prepare features for prediction
def get_input_array():
    vals = st.session_state.features
    return np.array([[
        vals['Sepal Length'],
        vals['Sepal Width'],
        vals['Petal Length'],
        vals['Petal Width']
    ]])

scaled = scaler.transform(get_input_array())

with result_col:
    st.subheader('Prediction')
    if st.button('Predict'):
        idx = model.predict(scaled)[0]
        species = iris.target_names[idx]
        st.success(f"#### Species:{species.upper()}")
        img_url = species_images.get(species)
        if img_url:
            st.image(img_url, caption=species.capitalize(), use_container_width=True)

# Final input measurements
st.subheader('Final Input Measurements')
final = pd.DataFrame(list(st.session_state.features.items()), columns=['Feature', 'Value'])
st.table(final)

st.caption('Sliders and table are synchronized. Prediction and image on the right.')
