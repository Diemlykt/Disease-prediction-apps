import requests

def main():
    try:
        response = requests.get("https://disease-prediction-apps.onrender.com/ping")
        print(f"Ping response status: {response.status_code}")
    except Exception as e:
        print(f"Ping failed: {e}")

if __name__ == "__main__":
    main()

