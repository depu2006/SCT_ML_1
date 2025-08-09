import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

print("🚀 Script started")
print("📄 Loading dataset...")

# ✅ Load dataset and drop Unnamed: 0 if it exists
df = pd.read_csv('Cleaned_data.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"✅ Dataset loaded. Shape: {df.shape}")

# Continue with preprocessing and model training
X = df.drop('price', axis=1)
y = df['price']

categorical_cols = ['location']
print("🧠 Categorical columns:", categorical_cols)

# Preprocessing
preprocessor = ColumnTransformer([
    ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("📊 Data split complete.")

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained.")

# Save the model and the preprocessor
pk.dump((model, preprocessor), open('House_prediction_model.pkl', 'wb'))

print("💾 Model saved to House_prediction_model.pkl")
