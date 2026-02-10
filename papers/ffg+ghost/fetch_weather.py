import requests
import json
from datetime import datetime

def get_weather_data(api_key, city):
    """
    Fetch current weather data for a given city.
    
    Args:
        api_key (str): OpenWeatherMap API key
        city (str): City name
    
    Returns:
        dict: Weather data or error information
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            return {
                'success': False,
                'error': f"API Error: {data.get('message', 'Unknown error')}"
            }
        
        processed_data = {
            'success': True,
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'clouds': data['clouds']['all'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
        }
        
        return processed_data
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Network error: {str(e)}"
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {
            'success': False,
            'error': f"Data parsing error: {str(e)}"
        }

def display_weather(weather_data):
    """
    Display weather data in a readable format.
    
    Args:
        weather_data (dict): Weather data from get_weather_data
    """
    if not weather_data.get('success'):
        print(f"Error: {weather_data.get('error', 'Unknown error')}")
        return
    
    print(f"\nWeather in {weather_data['city']}, {weather_data['country']}")
    print("=" * 40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s")
    if weather_data['wind_direction'] != 'N/A':
        print(f"Wind direction: {weather_data['wind_direction']}°")
    if weather_data['visibility'] != 'N/A':
        print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloud cover: {weather_data['clouds']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print(f"Last updated: {weather_data['timestamp']}")

if __name__ == "__main__":
    # Example usage
    API_KEY = "your_api_key_here"  # Replace with actual API key
    CITY = "London"
    
    print(f"Fetching weather data for {CITY}...")
    weather = get_weather_data(API_KEY, CITY)
    display_weather(weather)