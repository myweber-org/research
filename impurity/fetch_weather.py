
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city.
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set.")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            error_message = data.get("message", "Unknown error")
            raise Exception(f"API Error: {error_message}")

        main_data = data.get("main", {})
        weather_data = data.get("weather", [{}])[0]
        wind_data = data.get("wind", {})

        weather_info = {
            'city': data.get("name"),
            'country': data.get("sys", {}).get("country"),
            'temperature': main_data.get("temp"),
            'feels_like': main_data.get("feels_like"),
            'pressure': main_data.get("pressure"),
            'humidity': main_data.get("humidity"),
            'description': weather_data.get("description"),
            'wind_speed': wind_data.get("speed"),
            'wind_deg': wind_data.get("deg"),
            'timestamp': datetime.fromtimestamp(data.get("dt")).isoformat()
        }

        return weather_info

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}")

def display_weather(weather_info):
    """
    Display weather information in a readable format.
    """
    if not weather_info:
        print("No weather data available.")
        return

    print(f"Weather in {weather_info['city']}, {weather_info['country']}:")
    print(f"  Temperature: {weather_info['temperature']}°C (Feels like: {weather_info['feels_like']}°C)")
    print(f"  Conditions: {weather_info['description'].capitalize()}")
    print(f"  Humidity: {weather_info['humidity']}%")
    print(f"  Pressure: {weather_info['pressure']} hPa")
    print(f"  Wind: {weather_info['wind_speed']} m/s at {weather_info['wind_deg']}°")
    print(f"  Last updated: {weather_info['timestamp']}")

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty.")
            exit(1)

        weather = get_weather(city)
        display_weather(weather)

    except Exception as e:
        print(f"Error: {e}")