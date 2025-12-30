
import requests
import json
from datetime import datetime

def get_weather_data(api_key, city_name):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    Returns a dictionary containing temperature, humidity, and description.
    """
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
        
        if data['cod'] != 200:
            return {"error": f"API Error: {data.get('message', 'Unknown error')}"}
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather_description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"error": f"Data parsing error: {str(e)}"}

def display_weather(weather_data):
    """Display weather information in a readable format."""
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Conditions: {weather_data['weather_description'].title()}")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp']}")
    print("="*40)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    print(f"Fetching weather data for {CITY}...")
    weather = get_weather_data(API_KEY, CITY)
    display_weather(weather)