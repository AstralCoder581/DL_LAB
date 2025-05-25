import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load data or create dummy data
try:
    df = pd.read_csv('sonar_dataset.csv')
except FileNotFoundError:
    df = pd.DataFrame(np.random.randn(208, 61))
    df.columns = [f'f{i}' for i in range(60)] + ['class']
    df['class'] = np.random.choice(['R', 'M'], 208)

X = df.iloc[:, :-1].values
y = np.where(df['class'] == 'R', 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(dropout):
    model = models.Sequential()
    model.add(layers.Input(shape=(60,)))
    model.add(layers.Dense(128, activation='relu'))
    if dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    if dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

histories = {}
for label, do_dropout in [('With Dropout', True), ('Without Dropout', False)]:
    m = build_model(do_dropout)
    histories[label] = m.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test), verbose=0)
    print(f"{label} Test Accuracy: {m.evaluate(X_test, y_test, verbose=0)[1]:.3f}")

plt.figure(figsize=(10, 4))
for i, metric in enumerate(['accuracy', 'loss']):
    plt.subplot(1, 2, i+1)
    for label, history in histories.items():
        plt.plot(history.history[f'val_{metric}'], label=label)
    plt.title(f'Val {metric.title()}')
    plt.xlabel('Epoch')
    plt.legend()
plt.tight_layout()
plt.show()
