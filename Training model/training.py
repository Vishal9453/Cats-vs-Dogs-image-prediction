import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Load the preprocessed data from serialized files
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

# Normalize the pixel values to be in the range [0, 1]
X = X / 255

# Reshape the data to have a single channel (grayscale) and the appropriate dimensions
X = X.reshape(-1, 60, 60, 1)

# Create a Sequential model
model = Sequential()

# Add a convolutional layer with 64 filters and a 3x3 kernel, using ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add a max-pooling layer to downsample the spatial dimensions
model.add(MaxPooling2D((2, 2)))

# Add another convolutional layer with 64 filters and a 3x3 kernel, using ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another max-pooling layer
model.add(MaxPooling2D((2, 2)))

# Flatten the feature maps to prepare for the fully connected layers
model.add(Flatten())

# Add a fully connected (dense) layer with 128 units and ReLU activation
model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))

# Add the output layer with 2 units (one for each class) and softmax activation
model.add(Dense(2, activation='softmax'))

# Compile the model with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the preprocessed data for 5 epochs with a 10% validation split
model.fit(X, y, epochs=5, validation_split=0.1)
