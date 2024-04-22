import requests
import os

url = 'http://0.0.0.0:8000/predicting/json_prediction'

file_path = os.path.join(os.getcwd(), 'images', 'imagen.png')

with open(file_path, 'rb') as file:
    files = {'file': (file_path, file, 'image/png')} 
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Image uploaded")
        print("API result:", response.json())  # o response.json() si la respuesta es JSON
    else:
        print("Error status:", response.status_code)
