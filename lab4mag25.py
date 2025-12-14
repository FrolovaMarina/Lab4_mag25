import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv('Coffee_sales.csv')

df_res = df.copy()
encoder_coffee = LabelEncoder()
df_res['coffee_encode'] = encoder_coffee.fit_transform(df['coffee_name'])
features = ['hour_of_day', 'Weekdaysort', 'Monthsort', 'coffee_encode', 'money']
x = df_res[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

models = {
    #Метод К-средних  
    'K-Means': KMeans(n_init=10, n_clusters=3),
    #Иерархическая агломеративная кластеризация
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    #Гауссовская смесь
    'Gaussian Mixture': GaussianMixture(n_components=3),
    #Спектральная кластеризация
    'Spectral': SpectralClustering(n_clusters=3), 
    #BIRCH
    'BIRCH': Birch(n_clusters=3, threshold=0.3)
}

print("Результаты кластеризации:")

results = {}
cluster_labels = {}
 
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    cluster_labels[name] = labels  
    unique_labels = np.unique(labels)
    silhouette = silhouette_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    results[name] = {
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski_harabasz,
        'Davies-Bouldin': davies_bouldin
    }
    
    print(f"\n{name}:")    
    print(f"Метрика Silhouette: {silhouette:.3f}")
    print(f"Метрика Calinski-Harabasz: {calinski_harabasz:.2f}")
    print(f"Метрика Davies-Bouldin: {davies_bouldin:.3f}")

# График 1. Метрика Silhouette
plt.figure(figsize=(10, 6))
silhouette_res = [results[name]['Silhouette'] for name in results.keys()]
bars = plt.bar(results.keys(), silhouette_res, color=['#66CDAA', '#FFB6C1', '#6495ED', '#D8BFD8', '#008B8B'])
plt.title('Сравнение моделей по метрике Silhouette', fontweight='bold')
plt.ylabel('Value')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
             ha='center', va='bottom', fontweight='bold')

# График 2. Метрика Calinski-Harabasz
plt.figure(figsize=(10, 6))
calinskih_res = [results[name]['Calinski-Harabasz'] for name in results.keys()]
bars = plt.bar(results.keys(), calinskih_res, color=['#66CDAA', '#FFB6C1', '#6495ED', '#D8BFD8', '#008B8B'])
plt.title('Сравнение моделей по метрике Calinski-Harabasz', fontweight='bold')
plt.ylabel('Value')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
             ha='center', va='bottom', fontweight='bold')

# График 3. Метрика Davies-Bouldin
plt.figure(figsize=(10, 6))
daviesb_res = [results[name]['Davies-Bouldin'] for name in results.keys()]
bars = plt.bar(results.keys(), daviesb_res, color=['#66CDAA', '#FFB6C1', '#6495ED', '#D8BFD8', '#008B8B'])
plt.title('Сравнение моделей по метрике Davies-Bouldin)', fontweight='bold')
plt.ylabel('Value')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
             ha='center', va='bottom', fontweight='bold')

plt.show()


