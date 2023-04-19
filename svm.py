import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
raw_data = urllib.request.urlopen(url)
df = pd.read_csv(raw_data, sep=";")
y = np.where(data.target > 1, 1, 0)

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
C_range = [10 ** i for i in range(-2, 3)]
gamma_range = [10 ** i for i in range(-5, 0)]
degree_range = list(range(1, 6))

best_accuracy = 0
cvgs_data = []
all_samples = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sample_best_accuracy = 0
    sample_best_params = {}
    sample_cvgs_data = []

    for j in range(500):
        kernel = np.random.choice(kernel_list)
        C = np.random.choice(C_range)
        gamma = np.random.choice(gamma_range)
        degree = np.random.choice(degree_range) if kernel == 'poly' else None

        clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > sample_best_accuracy:
            sample_best_accuracy = accuracy
            sample_best_params = {'kernel': kernel, 'C': C, 'gamma': gamma, 'degree': degree}

        sample_cvgs_data.append(sample_best_accuracy)

    all_samples.append([sample_best_params['kernel'], sample_best_params['C'], sample_best_params['gamma'],
                        sample_best_params['degree'], sample_best_accuracy])

    if sample_best_accuracy > best_accuracy:
        cvgs_data = sample_cvgs_data
        best_accuracy = sample_best_accuracy

columns = ['Kernel', 'C', 'Gamma', 'Degree', 'Accuracy']
result_df = pd.DataFrame(all_samples, columns=columns)
print(result_df)
result_df.to_csv('./result.csv', index=False)
result_df.to_markdown('./result.md', index=False)

plt.plot(np.arange(len(cvgs_data)), cvgs_data)
plt.title('Convergence graph of best SVM')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()

