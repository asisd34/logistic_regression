import pandas as pd
 
# Veri setini yükle
data = pd.read_csv("winequality-red.csv")

# Veri setini incele
print(data.head())
print(data.info())

# Kaliteyi sınıflandır
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.5 else 0)

# Bağımlı değişkeni belirle
X = data.drop(columns=['quality'])  # Bağımsız değişkenler
Y = data['quality']  # Bağımlı değişken
 
# Veriyi eğitim ve test setlerine bölelim
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Ölçekleme işlemi uygulayalım
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistik regresyon modelini oluşturalım ve eğitelim
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Test seti üzerinde modelin performansını değerlendirelim
Y_pred = model.predict(X_test_scaled)
 
from sklearn.metrics import accuracy_score, confusion_matrix
 
# Doğruluk değerini hesaplayalım
accuracy = accuracy_score(Y_test, Y_pred)
print("Doğruluk Değeri:", accuracy)
 
# Karışıklık matrisini görselleştirelim
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Karışıklık Matrisi:\n", conf_matrix)
