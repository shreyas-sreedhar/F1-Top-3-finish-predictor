import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['top_3_finish'] = df['position'].apply(lambda x: 1 if x in ['1', '2', '3'] else 0)
    df['laps'].fillna(df['laps'].median(), inplace=True)
    df['car'].fillna(df['car'].mode()[0], inplace=True)
    return df

def encode_features(df):
    encoders = {
        'driver': LabelEncoder(),
        'car': LabelEncoder(),
        'grand_prix': LabelEncoder()
    }
    df['driver_encoded'] = encoders['driver'].fit_transform(df['driver'])
    df['car_encoded'] = encoders['car'].fit_transform(df['car'])
    df['grand_prix_encoded'] = encoders['grand_prix'].fit_transform(df['grand_prix'])
    return df, encoders

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load and preprocess data
f1_data = load_data('f1_data_races.csv')
f1_data = preprocess_data(f1_data)
f1_data, encoders = encode_features(f1_data)

# Train model
X = f1_data[['driver_encoded', 'car_encoded', 'grand_prix_encoded', 'laps']]
y = f1_data['top_3_finish']
model = train_model(X, y)

# Save model and encoders
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
for label, encoder in encoders.items():
    with open(f'{label}_encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
