import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetches current weather data for a given city.

    Args:
        city_name (str): Name of the city.
        api_key (str, optional): OpenWeatherMap API key. If not provided,
                                 tries to get from environment variable.

    Returns:
        dict: Weather data if successful, None otherwise.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            print("Error: API key not provided and OPENWEATHER_API_KEY environment variable not set.")
            return None

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None

        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind']['deg'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        return None

def display_weather(weather_data):
    """Prints formatted weather information."""
    if weather_data is None:
        print("No weather data to display.")
        return

    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Example usage
    city = "London"
    print(f"Fetching weather for {city}...")
    
    # For actual use, set OPENWEATHER_API_KEY environment variable
    # or pass api_key directly to get_weather()
    weather = get_weather(city)
    
    if weather:
        display_weather(weather)
    else:
        print("Failed to retrieve weather data.")