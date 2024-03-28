import keras

class NNModel(keras.Model):
    def __init__(self, feature_length):
        super().__init__()

        self.dense1 = keras.layers.Dense(64, input_shape=(feature_length,), activation="relu")
        self.dense2 = keras.layers.Dense(32, activation="relu")
        self.dense3 = keras.layers.Dense(32, activation="relu")
        self.dense4 = keras.layers.Dense(16, activation="relu")
        self.dense5 = keras.layers.Dense(16, activation="relu")
        self.dense6 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x= self.dense4(x)
        x= self.dense5(x)
        return self.dense6(x)

