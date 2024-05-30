# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('drone_model.pkl', 'rb') as f:
    preprocessor, pca, knn, target = pickle.load(f)

# Streamlit interface
st.title("Drone Recommender System")

st.write("""
### Enter the following parameters to get the best drone recommendations
""")

# User input parameters
flight_radius = st.slider("Flight Radius (meters)", 0, 10000, 5000)
flight_height = st.slider("Flight Height (meters)", 0, 10000, 5000)
cost = st.slider("Cost (USD)", 0, 15000, 7500)
battery_life = st.slider("Battery Life (minutes)", 0, 100, 50)
wind_resistance = st.slider("Wind Resistance (km/h)", 0, 100, 50)
payload_capacity = st.slider("Payload Capacity (kg)", 0.0, 20.0, 10.0)
noise_level = st.slider("Noise Level (dB)", 0, 100, 50)
regulatory_compliance = st.selectbox("Regulatory Compliance", [0, 1])
camera_quality = st.slider("Camera Quality (MP)", 0, 50, 25)
obstacle_avoidance = st.selectbox("Obstacle Avoidance", [0, 1])
user_rating = st.slider("User Rating", 1, 10, 5)

# Gather user input into a dataframe
user_data = pd.DataFrame([{
    'flight_radius': flight_radius,
    'flight_height': flight_height,
    'cost': cost,
    'battery_life': battery_life,
    'wind_resistance': wind_resistance,
    'payload_capacity': payload_capacity,
    'noise_level': noise_level,
    'regulatory_compliance': regulatory_compliance,
    'camera_quality': camera_quality,
    'obstacle_avoidance': obstacle_avoidance,
    'user_rating': user_rating
}])

if st.button("Recommend Drones"):
    # Preprocess the user input in the same way as the training data
    preprocessed_input = preprocessor.transform(user_data)
    pca_input = pca.transform(preprocessed_input)

    # Find the nearest drones
    distances, indices = knn.kneighbors(pca_input)

    best_drones = [target[i] for i in indices[0]]

    # Display the recommendations
    st.write("### Recommended Drones")
    for i, drone in enumerate(best_drones):
        st.write(f"{i + 1}. {drone}")
