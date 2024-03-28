import keras
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional, Dropout


class LSTMModel(keras.Model):
    def __init__(self, LSTM_units, n_components):
        super().__init__()
        self.lstm1 = Bidirectional(LSTM(units=LSTM_units, 
               input_shape=(None, n_components), 
               return_sequences=True))
        self.dropout1 = Dropout(0.1)
        self.lstm2 = Bidirectional(LSTM(units=LSTM_units, return_sequences=True))
        self.dropout2 = Dropout(0.1)
        self.lstm3 = Bidirectional(LSTM(units=LSTM_units, return_sequences=True))
        self.dropout3 = Dropout(0.1)
        self.time_distributed = TimeDistributed(Dense(1, activation='sigmoid'))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.lstm3(x)
        x = self.dropout3(x)
        return self.time_distributed(x)