
import requests
import json
from datetime import datetime

def get_weather(api_key, city_name, units='metric'):
    """
    Fetch current weather data for a given city.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': units
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return parse_weather_data(data)
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        return None

def parse_weather_data(data):
    """
    Parse and extract relevant weather information from API response.
    """
    main = data.get('main', {})
    weather = data.get('weather', [{}])[0]
    wind = data.get('wind', {})
    sys = data.get('sys', {})
    
    parsed_data = {
        'location': data.get('name'),
        'country': sys.get('country'),
        'temperature': main.get('temp'),
        'feels_like': main.get('feels_like'),
        'humidity': main.get('humidity'),
        'pressure': main.get('pressure'),
        'weather_main': weather.get('main'),
        'weather_description': weather.get('description'),
        'wind_speed': wind.get('speed'),
        'wind_direction': wind.get('deg'),
        'sunrise': datetime.fromtimestamp(sys.get('sunrise')).strftime('%H:%M:%S'),
        'sunset': datetime.fromtimestamp(sys.get('sunset')).strftime('%H:%M:%S'),
        'timestamp': datetime.fromtimestamp(data.get('dt')).isoformat()
    }
    
    return parsed_data

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['location']}, {weather_data['country']}")
    print("="*40)
    print(f"Condition: {weather_data['weather_main']} ({weather_data['weather_description']})")
    print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*40)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)