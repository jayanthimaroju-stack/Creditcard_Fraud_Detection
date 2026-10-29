import pandas as pd
import requests

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Select one transaction
transaction = df.drop(columns=["Class"]).iloc[0].tolist()
transaction = [float(value) for value in transaction]

# Send the transaction to the Flask API
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"features": transaction}
)

# Display the result
print("Status Code:", response.status_code)
print("API Response:", response.json())