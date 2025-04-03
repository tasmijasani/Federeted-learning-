import flwr as fl
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal


def load_data():
    pe_header = pd.read_csv("PEHeader.csv")
    cleaned_data = pd.read_csv("cleaned_data.csv")
    
    columns_to_keep = [
        'Machine', 'SizeOfOptionalHeader', 'Characteristics', 'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode',
        'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'ImageBase',
        'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion', 'MinorOperatingSystemVersion',
        'MajorImageVersion', 'MinorImageVersion', 'MajorSubsystemVersion', 'MinorSubsystemVersion',
        'SizeOfImage', 'SizeOfHeaders', 'CheckSum', 'Subsystem', 'DllCharacteristics',
        'SizeOfStackReserve', 'SizeOfHeapReserve', 'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes'
    ]
    columns_to_keep2 = columns_to_keep + ['Malware_Type']
    
    # Keep only necessary columns
    filtered_cleaned_data = cleaned_data[cleaned_data["legitimate"] == 0].copy()[columns_to_keep]
    filtered_cleaned_data["Malware_Type"] = "No_malware"
    
    pe_cleaned_header = pe_header[columns_to_keep2].copy()
    data = pd.concat([pe_cleaned_header, filtered_cleaned_data], ignore_index=True)

    # Separate features (X) and labels (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Encode string labels to integers
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    return X_train, X_test, y_train, y_test, num_classes   
    
def create_model(input_shape,num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=HeNormal()),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=HeNormal()),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=HeNormal())        
    ])
    return model

class ECGClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"📊 Round completed — Test Accuracy: {accuracy:.4f}")
        return loss, len(self.X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    X_train, X_test, y_train, y_test,num_classes = load_data()
    model = create_model(X_train.shape[1],num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    client = ECGClient(model, X_train, y_train, X_test, y_test)
    #fl.client.start_numpy_client(server_address="172.31.6.145:8080", client=client)
    fl.client.start_client(server_address="172.31.6.145:8080", client=client.to_client())

