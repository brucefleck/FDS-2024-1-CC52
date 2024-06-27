import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('RUvideos_cc50_202101.csv')
# Mostrar las primeras filas del DataFrame
#df.head()

# Ver la estructura del DataFrame
df.info()

# Ver las estadísticas descriptivas
print("Estadísticas descriptivas")
print(df.describe(include='all'))

# Ver los nombres de las columnas
print("Columnas")
print(df.columns)

# Ver la cantidad de valores nulos
print("Valores nulos")
print(df.isnull().sum())

# Tabla de frecuencia para category_id
print("Tabla de frecuencia")
print(df['category_id'].value_counts())

# Visualizar los datos
# Gráfico de barras de las categorías
plt.figure(figsize=(10,6))
sns.countplot(y='category_id', data=df, palette='viridis')
plt.title('Distribución de categorías')
plt.xlabel('Cantidad')
plt.ylabel('ID de Categoría')
plt.legend(['Cantidad de videos por categoría'])
plt.show()

# Gráfico de dispersión para 'views' y 'likes')
plt.figure(figsize=(10,6))
sns.scatterplot(x='views', y='likes', data=df, alpha=0.6)
plt.title('Relación entre Vistas y Me gusta')
plt.xlabel('Vistas')
plt.ylabel('Me gusta')
plt.legend(['Vistas vs. Me gusta'])
plt.show()

# Eliminar columnas irrelevantes
df.drop(['thumbnail_link', 'description', 'video_error_or_removed'], axis=1, inplace=True)

# Eliminar filas con valores nulos
df['tags'].fillna('', inplace=True)
df.dropna(subset=['lat', 'lon'], inplace=True)  

# Convertir 'publish_time' a datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], format='%d/%m/%Y %H:%M')

# Normalización de los datos numéricos
scaler = StandardScaler()
numerical_cols = ['views', 'likes', 'dislikes', 'comment_count', 'lat', 'lon']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Codificación de variables categóricas
df = pd.get_dummies(df, columns=['category_id', 'comments_disabled', 'ratings_disabled', 'state'], drop_first=True)


# distribución de las variables numéricas después de la normalización
plt.figure(figsize=(15,10))
df[numerical_cols].hist(bins=30, figsize=(15, 10), layout=(3, 2))
plt.suptitle('Distribución de las variables numéricas después de la normalización')
plt.show()

# Calcular la matriz de correlación
numeric_columns = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()
