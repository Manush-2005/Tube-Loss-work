<<<<<<< HEAD
import time
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def quantile_loss(q):
    def loss(y_true, y_pred):
        y = y_true[:, 0]
        e = y - y_pred[:, 0]
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss


class ANNQuantileReg:
    def __init__(
        self,
        input_dim=1,
        hidden_dim=200,
        q=0.5,
        lr=0.02
    ):
        self.q = q
        self.model = self._build_model(input_dim, hidden_dim)
        self._compile(lr)

    def _build_model(self, input_dim, hidden_dim):
        model = Sequential()
        model.add(Dense(
            hidden_dim,
            input_dim=input_dim,
            activation='relu',
            kernel_initializer=keras.initializers.RandomNormal(0.0, 0.2)
        ))
        model.add(Dense(
            hidden_dim,
            activation='relu',
            kernel_initializer=keras.initializers.RandomNormal(0.0, 0.2)
        ))
        model.add(Dense(
            1,  # SINGLE quantile output
            activation='linear',
            kernel_initializer=keras.initializers.RandomNormal(0.0, 0.3),
            bias_initializer=keras.initializers.Constant(0.0)
        ))
        return model

    def _compile(self, lr):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=0.01
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(
            optimizer=opt,
            loss=quantile_loss(self.q)
        )

    def fit(self, X, y, epochs=400, batch_size=40, verbose=0):
        start = time.time()
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        self.train_time = time.time() - start
        return self.train_time

    def predict(self, X):
        return self.model.predict(X, verbose=0)[:, 0]
=======
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNQuantileReg(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=200):
        super().__init__()
        
       
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  
        
    
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.2)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.fc2.bias, torch.tensor([-3.0, 3.0]))
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
>>>>>>> af3ea825c59c75cbac2452e426fa418b60a8d2ee
