import requests

#Token de la API de INEGÍ
token = "e5237bf2-3013-a9ae-d0e2-bca800cefa52"

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_data(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

if __name__ == "__main__":
    base_url = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR"
    client = APIClient(base_url)
    
    try:
        endpoint = f"1002000041/es/0700/false/BISE/2.0/{token}?type=json"
        data = client.get_data(endpoint, params={"key": "value"})
        print(data)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")