from echo import optimize
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD


def optimize_model(train_data_np, train_labels_np, val_data_np, val_labels_np, test_data_np, test_labels_np,
                   model_type='random_forest'):
    """
    Function to optimize hyperparameters for either a Random Forest or a CNN.
    :param train_data_np: ndarray for training data
    :param train_labels_np: ndarray for training labels
    :param val_data_np: ndarray for validation data
    :param val_labels_np: ndarray for validation labels
    :param test_data_np: ndarray for testing data
    :param test_labels_np: ndarray for testing labels
    :param model_type: string, either 'random_forest' or 'cnn'
    :return: best model with optimized hyperparameters
    """

    if model_type == 'random_forest':
        # Define hyperparameter search space for Random Forest
        search_space = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Create the optimization task for Random Forest
        opt_task = optimize.HyperparameterOptimization(
            model=RandomForestClassifier,
            param_space=search_space,
            scoring='accuracy',
            X_train=train_data_np,
            y_train=train_labels_np
        )

    elif model_type == 'cnn':
        # Assuming the input shape is (nsamples, pixel_height, pixel_width, 1) for CNN

        # Define hyperparameter search space for CNN
        search_space = {
            'filters': [32, 64],
            'kernel_size': [(3, 3), (5, 5)],
            'pool_size': [(2, 2)],
            'optimizer': ['adam', 'sgd'],
            'learning_rate': [0.001, 0.01]
        }

        def create_cnn_model(trial):
            # Hyperparameters to tune
            filters = trial.suggest_categorical('filters', [32, 64])
            kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (5, 5)])
            pool_size = trial.suggest_categorical('pool_size', [(2, 2)])
            optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

            # Create a CNN model
            model = Sequential()
            model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=train_data_np.shape[1:]))
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(train_labels_np.shape[1], activation='softmax'))  # Assuming one-hot encoded labels

            if optimizer == 'adam':
                opt = Adam(learning_rate=learning_rate)
            else:
                opt = SGD(learning_rate=learning_rate)

            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        # Create the optimization task for CNN
        opt_task = optimize.HyperparameterOptimization(
            model=create_cnn_model,
            param_space=search_space,
            scoring='accuracy',
            X_train=train_data_np,
            y_train=train_labels_np
        )

    else:
        raise ValueError("Invalid model type. Choose either 'random_forest' or 'cnn'.")

    # Run the optimization
    best_params = opt_task.run()
    print(f"Best Parameters for {model_type}: {best_params}")

    if model_type == 'random_forest':
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(train_data_np, train_labels_np)
        accuracy = best_model.score(test_data_np, test_labels_np)

    elif model_type == 'cnn':
        best_model = create_cnn_model(best_params)
        best_model.fit(train_data_np, train_labels_np, epochs=10, batch_size=32,
                       validation_data=(val_data_np, val_labels_np), verbose=0)
        accuracy = best_model.evaluate(test_data_np, test_labels_np, verbose=0)[1]

    print(f"Optimized {model_type} Model Accuracy: {accuracy}")
    return best_model

# Example usage:
# For Random Forest
# optimize_model(train_data_np, train_labels_np, val_data_np, val_labels_np, test_data_np, test_labels_np, 'random_forest')

# For CNN
# optimize_model(train_data_np, train_labels_np, val_data_np, val_labels_np, test_data_np, test_labels_np, 'cnn')
