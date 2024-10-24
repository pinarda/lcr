from echo.src.base_objective import BaseObjective
from echo.src.pruners import KerasPruningCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Generate sample data: 1000 samples, 20 features
X = np.random.rand(1000, 20)
y = np.random.randint(0, 10, size=(1000,))  # 10 classes

# Convert labels to categorical (one-hot encoding)
y_categorical = to_categorical(y, num_classes=10)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Display shapes of the data
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)


# Custom function to update the configuration with trial hyperparameters
def custom_updates(trial, conf):
    # Update configuration for learning rate and dropout based on trial suggestions
    conf['optimizer']['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    conf['model']['dropout'] = trial.suggest_uniform('dropout', 0.0, 0.5)
    return conf

# Define the Objective class that uses the BaseObjective from echo-opt
class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):
        BaseObjective.__init__(self, config, metric)

    def build_model(self, conf):
        # Example Keras model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(conf['input_shape'],)))
        model.add(Dropout(conf['model']['dropout']))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(10, activation='softmax'))  # Adjust for the number of classes in your dataset

        # Set optimizer based on the configuration (custom_updates)
        if conf['optimizer']['type'] == 'adam':
            optimizer = Adam(learning_rate=conf['optimizer']['learning_rate'])
        else:
            optimizer = SGD(learning_rate=conf['optimizer']['learning_rate'])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, trial, conf):
        # Make custom updates to the model configuration based on the hyperparameters suggested by the trial
        conf = custom_updates(trial, conf)

        # Build the model
        model = self.build_model(conf)

        # Load data (you should replace this with your dataset loading logic)
        # X_train, y_train, X_val, y_val = load_your_data()

        callbacks = [KerasPruningCallback(trial, self.metric, interval=1)]
        result = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=callbacks)

        # Return the evaluation metrics
        results_dictionary = {
            "val_loss": result.history['val_loss'][-1],
            "val_accuracy": result.history['val_accuracy'][-1]
        }
        return results_dictionary
