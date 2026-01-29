
import requests
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str): OpenWeatherMap API key. If None, uses environment variable.
    
    Returns:
        dict: Weather data including temperature, humidity, and description
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        raise ValueError("API key must be provided or set in OPENWEATHER_API_KEY environment variable")
    
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
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather_description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'visibility': data.get('visibility', 'N/A'),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch weather data: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Unexpected API response format: {str(e)}")

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data dictionary from get_weather function
    """
    if not weather_data:
        print("No weather data available")
        return
    
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
    print(f"Time: {weather_data['timestamp']}")
    print("-" * 40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather_description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")

if __name__ == "__main__":
    # Example usage
    try:
        # For testing, you can set your API key here or use environment variable
        # api_key = "your_api_key_here"
        api_key = None  # Will use environment variable
        
        city = "London"
        weather = get_weather(city, api_key)
        display_weather(weather)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have a valid OpenWeatherMap API key")
        print("Set it as environment variable: export OPENWEATHER_API_KEY='your_key'")