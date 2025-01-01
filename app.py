import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# App Title
st.title('Penguin Species Prediction')
st.info('This app helps predict the species of a penguin based on its physical features.')

# Data Section
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.dataframe(df)

    st.write('**Feature Matrix (X)**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**Target Variable (y)**')
    y_raw = df['species']
    st.dataframe(y_raw)

# Data Visualization
with st.expander('Data Visualization'):
    st.write('Scatter Plot of Bill Length vs Body Mass')
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar Input Features
with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

    # Create DataFrame for Input Features
    input_features = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(input_features, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# Display Input Features
with st.expander('Input Features'):
    st.write('**Input Penguin Data**')
    st.dataframe(input_df)
    st.write('**Combined Penguins Data (For Encoding)**')
    st.dataframe(input_penguins)

# Data Preparation
encode_cols = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode_cols)

X = df_penguins[1:]  # Exclude the first row (input penguin data)
input_row = df_penguins[:1]  # Input penguin row for prediction

# Encode Target Variable
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    return target_mapper[val]


y = y_raw.apply(target_encode)

# Display Encoded Data
with st.expander('Data Preparation'):
    st.write('**Encoded Feature Matrix (X)**')
    st.dataframe(input_row)
    st.write('**Encoded Target Variable (y)**')
    st.dataframe(y)

# Model Training and Prediction
clf = RandomForestClassifier()
clf.fit(X, y)

# Prediction
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Prepare Prediction DataFrame
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display Prediction Results
st.subheader('Predicted Species')
st.dataframe(
    df_prediction_proba,
    column_config={
        'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
        'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
        'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1),
    },
    hide_index=True
)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f'Predicted Species: {penguins_species[prediction[0]]}')
