import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#file_path = "soil_weather_crop_data.csv" add a huge csv dataset if you want

try:
    df = pd.read_csv(file_path)
    print("CSV file loaded successfully bro. Here are the first few rows lol:")
    print(df.head())
except FileNotFoundError:
    print("File not found bye bye.")
    exit()

numeric_columns = ['soil_ph', 'rainfall', 'temperature', 'humidity']


X_numeric = df[numeric_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Manually create labels for the crops based on hypothetical rules
conditions = [
    (df['soil_ph'] < 6.0) & (df['rainfall'] > 1000) & (df['temperature'] < 30),#rice
    (df['soil_ph'] >= 6.0) & (df['soil_ph'] <= 7.0) & (df['rainfall'] < 1000) & (df['temperature'] < 25), #wheat
    (df['soil_ph'] > 7.0) & (df['rainfall'] > 500) & (df['temperature'] >= 25) #sugarcane
]

choices = [0, 1, 2]  # 0 for rice, 1 for wheat, 2 for sugarcane

y = np.select(conditions, choices, default=0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Build and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)


# Function to preprocess input data
def preprocess_input(soil_ph, rainfall, temperature, humidity, scaler):
    # Scale the input data
    input_data = np.array([[soil_ph, rainfall, temperature, humidity]])
    input_scaled = scaler.transform(input_data)
    return input_scaled


# Function to predict crop based on input data
def predict_crop(soil_ph, rainfall, temperature, humidity, model, scaler):
    # Preprocess the input
    input_scaled = preprocess_input(soil_ph, rainfall, temperature, humidity, scaler)

    # Predict the probabilities of each class
    predictions = model.predict(input_scaled)

    # Get the predicted class index
    prediction = np.argmax(predictions, axis=-1)[0]

    # Convert numerical prediction to crop name
    crop_names = {0: 'rice', 1: 'wheat', 2: 'sugarcane'}
    predicted_crop = crop_names[prediction]

    return predicted_crop

input_soil_ph = input("Enter the soil ph: ")
input_rainfall = input("Enter the rainfall: ")
input_temperature = input("Enter the temperature: ")
input_humidity = input("Enter the humidity: ")

predicted_crop = predict_crop(input_soil_ph, input_rainfall, input_temperature, input_humidity, model, scaler)
print(
    f'Based on input values: Soil pH={input_soil_ph}, Rainfall={input_rainfall}, Temperature={input_temperature}, Humidity={input_humidity}')
print(f'The predicted crop is: {predicted_crop}')

