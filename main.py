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

# 1. ¿Qué categorías de videos son las de mayor tendencia?
tendencia_por_categoria = df.groupby('category_id')['trending_date'].count().sort_values(ascending=False)
categorias_mas_tendencia = tendencia_por_categoria.head(5) 

# Graficar
plt.figure(figsize=(10, 6))
categorias_mas_tendencia.plot(kind='bar', color='skyblue')
plt.title('Categorías de videos con mayor tendencia')
plt.xlabel('ID de Categoría')
plt.ylabel('Cantidad de videos tendenciales')
plt.xticks(rotation=45)
plt.show()

# 2. ¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan?
likes_por_categoria = df.groupby('category_id')['likes'].sum().sort_values(ascending=False)
categorias_mas_gustan = likes_por_categoria.head(5)
categorias_menos_gustan = likes_por_categoria.tail(5)

# Graficar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de categorías con más "Me gusta"
categorias_mas_gustan.plot(kind='bar', color='lightgreen', ax=ax1)
ax1.set_title('Categorías de videos con más "Me gusta"')
ax1.set_xlabel('ID de Categoría')
ax1.set_ylabel('Total de "Me gusta"')
ax1.tick_params(axis='x', rotation=0)

# Añadir etiquetas de valor
for i, v in enumerate(categorias_mas_gustan):
    ax1.text(i, v, f'{v:,.0f}', ha='center', va='bottom')

# Gráfico de categorías con menos "Me gusta"
categorias_menos_gustan.plot(kind='bar', color='salmon', ax=ax2)
ax2.set_title('Categorías de videos con menos "Me gusta"')
ax2.set_xlabel('ID de Categoría')
ax2.set_ylabel('Total de "Me gusta"')
ax2.tick_params(axis='x', rotation=0)

# Añadir etiquetas de valor
for i, v in enumerate(categorias_menos_gustan):
    ax2.text(i, v, f'{v:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 3. ¿Qué categorías de videos tienen la mejor proporción (ratio) de "Me gusta" / "No me gusta"?
ratio_likes_dislikes = df.groupby('category_id')[['likes', 'dislikes']].sum()
ratio_likes_dislikes['ratio_likes_dislikes'] = ratio_likes_dislikes['likes'] / (ratio_likes_dislikes['dislikes'] + 1)

# Ordenar por ratio de mayor a menor
categorias_mejor_ratio = ratio_likes_dislikes.sort_values(by='ratio_likes_dislikes', ascending=False).head(5)

# Graficar
plt.figure(figsize=(12, 6))
ax = categorias_mejor_ratio['ratio_likes_dislikes'].plot(kind='bar', color='lightblue')
plt.title('Categorías de videos con mejor ratio de "Me gusta" / "No me gusta"')
plt.xlabel('ID de Categoría')
plt.ylabel('Ratio "Me gusta" / "No me gusta"')
plt.xticks(rotation=0)

# Añadir etiquetas de valor encima de cada barra
for i, v in enumerate(categorias_mejor_ratio['ratio_likes_dislikes']):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# 4. ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”?
ratio_views_comments = df.groupby('category_id')[['views', 'comment_count']].sum()
ratio_views_comments['ratio_views_comments'] = ratio_views_comments['views'] / (ratio_views_comments['comment_count'] + 1)

# Ordenar por ratio de mayor a menor
categorias_mejor_ratio_views_comments = ratio_views_comments.sort_values(by='ratio_views_comments', ascending=False).head(5)

# Graficar
plt.figure(figsize=(10, 6))
categorias_mejor_ratio_views_comments['ratio_views_comments'].plot(kind='bar', color='lightcoral')
plt.title('Categorías de videos con mejor ratio de "Vistas" / "Comentarios"')
plt.xlabel('ID de Categoría')
plt.ylabel('Ratio "Vistas" / "Comentarios"')
plt.xticks(rotation=45)
plt.show()

#5. ¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo? 
# Agrupar por fecha y contar la cantidad de videos en tendencia por día
videos_en_tendencia_por_dia = df.groupby('trending_date').size()

# Graficar
plt.figure(figsize=(12, 6))
videos_en_tendencia_por_dia.plot(color='skyblue')
plt.title('Volumen de videos en tendencia a lo largo del tiempo')
plt.xlabel('Fecha(Año/Dia/Mes)')
plt.ylabel('Cantidad de videos en tendencia')
plt.grid(True)
plt.show()

#6. ¿Qué canales de YouTube son tendencia más frecuentemente? ¿Y cuáles con menos frecuencia?
# Contar la frecuencia de tendencias por canal
canales_frecuencia = df['channel_title'].value_counts()

# Mostrar los canales más frecuentes
print("Canales más frecuentes en tendencia:")
print(canales_frecuencia.head(10))

# Mostrar los canales menos frecuentes
print("\nCanales menos frecuentes en tendencia:")
canales_menos_frecuentes = canales_frecuencia[canales_frecuencia == 1]
print(canales_menos_frecuentes.head(10))

#Grafica de canales más frecuentes
plt.figure(figsize=(12, 6))
canales_frecuencia.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Canales de YouTube en Tendencia')
plt.xlabel('Canales')
plt.ylabel('Número de tendencias')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#7. ¿En qué Estados se presenta el mayor número de "Vistas", "Me gusta" y "No me gusta"? 
# Agrupar por Estado y sumar las vistas, me gusta y no me gusta
estados_datos = df.groupby('state')[['views', 'likes', 'dislikes']].sum().reset_index()

# Funciones para formatear números grandes
def format_num(num):
    return f'{num:,.0f}'

# Imprimir resultados
print("Estados con mayor número de Vistas:")
print(estados_datos.sort_values(by='views', ascending=False).head(5).to_string(index=False, formatters={'views': format_num}))
print("\nEstados con mayor número de Me gusta:")
print(estados_datos.sort_values(by='likes', ascending=False).head(5).to_string(index=False, formatters={'likes': format_num}))
print("\nEstados con mayor número de No me gusta:")
print(estados_datos.sort_values(by='dislikes', ascending=False).head(5).to_string(index=False, formatters={'dislikes': format_num}))

# Función para crear gráficos
def plot_top_states(data, column, title, color):
    plt.figure(figsize=(10, 6))
    ax = data.sort_values(by=column, ascending=False).head(10).plot(kind='barh', x='state', y=column, color=color)
    plt.title(title)
    plt.xlabel(f'Número de {column.capitalize()}')
    plt.ylabel('Estado')
    plt.gca().invert_yaxis()
    
    # Añadir etiquetas de valor
    for i, v in enumerate(data.sort_values(by=column, ascending=False).head(10)[column]):
        ax.text(v, i, f' {format_num(v)}', va='center')
    
    plt.tight_layout()
    plt.show()

# Crear gráficos
plot_top_states(estados_datos, 'views', 'Top 10 Estados por Número de Vistas', 'skyblue')
plot_top_states(estados_datos, 'likes', 'Top 10 Estados por Número de Me gusta', 'lightgreen')
plot_top_states(estados_datos, 'dislikes', 'Top 10 Estados por Número de No me gusta', 'lightcoral')
