from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbours import KNeighboursClassifiers
from sklearn.model_selection import train_test_split , cross_val_score 
import matplotlib.pyplot as plt
data = load_iris()
# print(data)
X = data.data
Y = data.target
X_train, X_test,y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
k_range = range(1,31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"K={k}, CV Scores: {scores}, Mean CV Scores: {scores.mean()}")
    cv_scores.append(scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='*', color = 'green')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding Optimal K (Bias-Variance Tradeoff)')
# plt.grid(True)
plt.show()