import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Baca data
df = pd.read_csv('heart.csv', sep=';')
print("Contoh data:")
print(df.head())

# Pisahkan fitur dan target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ----------------------------
# Decision Tree Classifier
# ----------------------------
print("\n=== Decision Tree ===")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=0)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Classification Report (DT):")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=dt_model.classes_)
disp_dt.plot(cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# ----------------------------
# K-Nearest Neighbor (k=5)
# ----------------------------
print("\n=== KNN (k=5) ===")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("Classification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn_model.classes_)
disp_knn.plot(cmap='Greens')
plt.title("Confusion Matrix - KNN")
plt.show()

# ----------------------------
# Visualisasi Akurasi untuk nilai k
# ----------------------------
from sklearn.metrics import accuracy_score

k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"k = {k}, Accuracy = {acc:.4f}")

plt.plot(k_values, accuracies, marker='o')
plt.title("Akurasi vs Nilai k (KNN)")
plt.xlabel("Nilai k")
plt.ylabel("Akurasi")
plt.grid(True)
plt.show()
