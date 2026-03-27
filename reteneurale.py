import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

print(f"TensorFlow Version: {tf.__version__}")

X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 0.2, X.shape)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Dati generati', alpha=0.7)
plt.xlabel('Input (X)')
plt.ylabel('Output (y)')
plt.title('Dataset Sintetico per Regressione Lineare')
plt.grid(True)
plt.legend()
plt.show()

model = keras.Sequential([
    layers.Input(shape=(1,)), 
    layers.Dense(units=10, activation='tanh', name='hidden_layer_1'), 
    layers.Dense(units=1, name='output_layer')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

history = model.fit(
    X, y,
    epochs=200, 
    batch_size=32, 
    verbose=0 
)

history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss', 'mean_absolute_error']].plot()
plt.xlabel('Epoche')
plt.ylabel('Valore')
plt.title('Funzione di Loss e MAE durante l\'Addestramento (200 Epoche) con Tanh')
plt.grid(True)
plt.show()

y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Dati reali', alpha=0.7)
plt.plot(X, y_pred, color='red', label='Predizioni del modello', linewidth=2)
plt.xlabel('Input (X)')
plt.ylabel('Output (y)')
plt.title('Predizioni del Modello vs. Dati Reali (200 Epoche) con Tanh')
plt.grid(True)
plt.legend()
plt.show()

loss, mae = model.evaluate(X, y, verbose=0)
print(f"Loss finale (MSE): {loss:.4f}")
print(f"Errore Medio Assoluto (MAE) finale: {mae:.4f}")
