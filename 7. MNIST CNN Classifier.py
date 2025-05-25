import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class MNISTCNN:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self._preprocess()
        self.model = self._build_model()
    
    def _preprocess(self):
        # Normalize and reshape in 1 step
        self.x_train = self.x_train[..., np.newaxis] / 255.0
        self.x_test = self.x_test[..., np.newaxis] / 255.0
        # One-hot encode labels
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)
    
    def _build_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    
    def train(self, epochs=5):
        self.model.compile(optimizer='adam', 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
        return self.model.fit(self.x_train, self.y_train, 
                             validation_data=(self.x_test, self.y_test),
                             epochs=epochs, batch_size=128)
    
    def show_predictions(self, num=5):
        samples = self.x_test[:num]
        preds = np.argmax(self.model.predict(samples), axis=1)
        plt.figure(figsize=(10, 2))
        for i in range(num):
            plt.subplot(1, num, i+1)
            plt.imshow(samples[i], cmap='gray')
            plt.title(f"Pred: {preds[i]}")
            plt.axis('off')
        plt.show()

# Usage
cnn = MNISTCNN()
history = cnn.train()
cnn.show_predictions()
