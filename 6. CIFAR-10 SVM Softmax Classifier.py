import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# Prepare data for SVM (flatten images and use subsets)
x_train_svm, y_train_svm = x_train.reshape(len(x_train), -1)[:5000], y_train[:5000]
x_test_svm, y_test_svm = x_test.reshape(len(x_test), -1)[:1000], y_test[:1000]

def svm_classifier():
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(x_train_svm, y_train_svm)
    y_pred = svm.predict(x_test_svm)
    acc = accuracy_score(y_test_svm, y_pred)
    print(f"SVM Accuracy: {acc:.4f}")
    return acc

def softmax_classifier():
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, batch_size=128, epochs=10, validation_data=(x_test, y_test_cat), verbose=1)
    _, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Softmax Accuracy: {acc:.4f}")
    return acc

# Run and compare classifiers
svm_acc = svm_classifier()
softmax_acc = softmax_classifier()
print(f"\nComparison:\nSVM Accuracy: {svm_acc:.4f}\nSoftmax Accuracy: {softmax_acc:.4f}")
