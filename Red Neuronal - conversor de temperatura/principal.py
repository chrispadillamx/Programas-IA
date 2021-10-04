#Red Neuronal para convertir de grados Celsius a Fahrenheit

#Importo tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
#Importo numpy para trabajar con números
import numpy as np

#Agrego las temperaturas celsius que tengo para la capa de entrada
celsius=np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

#Agrego las temperaturas fahrenheit que tengo para la capa de salida
fahrenheit=np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Además de tensorflow, se usará Keras para esta red neuronal, que permite hacer las redes neuronales de manera simple, ahorra muchas líneas de código

#Creo una capa de tipo densa (tiene conexión desde una neurona de una capa hacia todas las neuronas de la siguiente capa)
#Este ejemplo solamente tiene una neurona (units)
#Grados celcius --> Grados Fahrenheit
#Las capas de entrada se determinan mediante input_shape, aquí tenemos una solamente

capa=tf.keras.layers.Dense(units=1, input_shape=[1])

#Usarermos el  modelo de Keras de tipo Secuencial, que es muy sencillo y básico
modelo=tf.keras.Sequential([capa])

#Comienza el aprendizaje en la red neuronal
#El optmizador Adam permite a la red cómo ajustar los pesos y sesgos de la red de manera eficiente para que vaya mejorando, y dentro del mismo va la tasa de aprendizaje, en este caso es 0.1
#La tasa de pérdida se calculará mediante el error cuadrático medio [una poca cantidad de errores grandes es peor que una gran cantidad de errores pequeños]
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("comenzando entrenamiento...")
#Se determina la cantidad de épocas en el entrenamiento
historial=modelo.fit(celsius,fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("#Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])


#Hacemos una predicción
print("Hagamos una predicción")
resultado=modelo.predict([80])
print("El resultado es "+str(resultado)+"fahrenheit!")
