import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# 1. I'm setting up the data paths and labels
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['HELLO', 'THANKS'])
no_sequences = 30
sequence_length = 30

# 2. I'm creating a map for the labels
label_map = {label:num for num, label in enumerate(actions)}

# 3. I'm loading all the data from the folders
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Converting to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 4. Building the LSTM Neural Network
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compiling the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("I'm starting the training process. This will take about 1 minute...")

# 5. Training the model
model.fit(X_train, y_train, epochs=200)

model.summary()

# 6. Saving the trained 'brain'
model.save('action.keras')
print("Model trained and saved as 'action.keras'!")
