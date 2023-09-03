import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import TensorBoard
import time

# Load the preprocessed data from serialized files
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

# Normalize the pixel values to be in the range [0, 1]
X = X / 255

# Reshape the data to have a single channel (grayscale) and the appropriate dimensions
X = X.reshape(-1, 60, 60, 1)

# Define lists of hyperparameters to experiment with
dense_layers = [3]
conv_layers = [3]
neurons = [64]

# Iterate through different combinations of hyperparameters
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        for neuron in neurons:

            # Create a unique name for this combination based on hyperparameters and timestamp
            NAME = '{}-denselayer-{}-convlayer-{}-neuron-{}'.format(dense_layer, conv_layer, neuron, int(time.time()))
            
            # Create a TensorBoard callback for visualizing training
            tensorboard = TensorBoard(log_dir='logs2\\{}'.format(NAME))

            # Create a Sequential model
            model = Sequential()

            # Add convolutional layers with specified number of neurons and pooling
            for l in range(conv_layer):
                model.add(Conv2D(neuron, (3, 3), activation='relu'))
                model.add(MaxPooling2D((2, 2)))

            # Flatten the feature maps to prepare for the fully connected layers
            model.add(Flatten())

            # Add a fully connected (dense) layer with ReLU activation
            model.add(Dense(neuron, input_shape=X.shape[1:], activation='relu'))

            # Add additional dense layers based on dense_layer hyperparameter
            for l in range(dense_layer - 1):
                model.add(Dense(neuron, activation='relu'))

            # Add the output layer with 2 units (one for each class) and softmax activation
            model.add(Dense(2, activation='softmax'))

            # Compile the model with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            # Train the model on the preprocessed data with specified hyperparameters
            model.fit(X, y, epochs=8, batch_size=32, validation_split=0.1, callbacks=[tensorboard])

            # Save the trained model
            model.save('Cats_vs_Dogs.model')

            # Save the model using pickle (not recommended for large models)
            with open('model.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)
