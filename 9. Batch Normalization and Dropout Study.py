import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class BatchNormDropoutStudy:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)

    def build_model(self, use_batchnorm=False, use_dropout=False):
        model = models.Sequential()
        
        # Conv Block 1
        model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
        if use_batchnorm: model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(32, (3, 3)))
        if use_batchnorm: model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        if use_dropout: model.add(layers.Dropout(0.25))

        # Conv Block 2
        model.add(layers.Conv2D(64, (3, 3)))
        if use_batchnorm: model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3, 3)))
        if use_batchnorm: model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        if use_dropout: model.add(layers.Dropout(0.25))

        # Dense Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        if use_batchnorm: model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        if use_dropout: model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def compare_models(self, epochs=10):
        configs = {
            'Baseline': (False, False),
            'BatchNorm': (True, False),
            'Dropout': (False, True),
            'Both': (True, True)
        }
        
        histories = {}
        for name, (bn, do) in configs.items():
            print(f'\nTraining {name} model...')
            model = self.build_model(bn, do)
            histories[name] = model.fit(
                self.x_train, self.y_train,
                validation_data=(self.x_test, self.y_test),
                epochs=epochs, batch_size=32, verbose=0
            )
            _, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
            print(f'{name} Test Accuracy: {acc:.4f}')
        
        self.plot_results(histories)
        return histories

    def plot_results(self, histories):
        plt.figure(figsize=(15, 5))
        metrics = ['accuracy', 'val_accuracy', 'val_loss']
        titles = ['Training Accuracy', 'Validation Accuracy', 'Validation Loss']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 3, i+1)
            for name, history in histories.items():
                plt.plot(history.history[metric], label=name)
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(title.split()[-1])
            plt.legend()
            
        plt.tight_layout()
        plt.show()

# Usage
study = BatchNormDropoutStudy()
histories = study.compare_models(epochs=5)
