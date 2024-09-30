# Importar as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Carregar o dataset (Wine Quality Red Wine)
df = pd.read_csv("winequality-red.csv")

# 2. Análise inicial dos dados
print(df.info())
print(df.describe())

# 3. Pré-processamento dos dados
# Dividir os dados entre características (X) e a variável alvo (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Binarizando a variável alvo (classificação: bom >= 7)
y = y.apply(lambda x: 1 if x >= 7 else 0)

# Normalizando os dados de entrada (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Definir o modelo de Rede Neural Artificial
model = Sequential()

# Primeira camada oculta com 128 neurônios e função de ativação ReLU
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# Segunda camada oculta com 64 neurônios e função de ativação ReLU
model.add(Dense(64, activation='relu'))

# Camada de saída com 1 neurônio e função de ativação sigmoid (para classificação binária)
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Treinar o modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 6. Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

# 7. Visualizar as métricas (Perda e Acurácia durante o treinamento)
# Plotando a perda (loss)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda Durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotando a acurácia
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia Durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
