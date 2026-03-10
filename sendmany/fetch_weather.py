
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, cache_duration: int = 300):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = cache_duration

    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        cache_key = city.lower()
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache[cache_key] = {
                'data': weather_data,
                'timestamp': time.time()
            }
            
            return weather_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing weather data: {e}")
            return None

    def clear_cache(self) -> None:
        self.cache.clear()
        print("Weather cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self.cache),
            'cached_cities': list(self.cache.keys()),
            'cache_duration': self.cache_duration
        }

def format_weather_report(weather_data: Dict[str, Any]) -> str:
    if not weather_data:
        return "No weather data available"
    
    return f"""
Weather Report for {weather_data['city']}:
----------------------------------------
Temperature: {weather_data['temperature']}°C
Humidity: {weather_data['humidity']}%
Conditions: {weather_data['description'].title()}
Wind Speed: {weather_data['wind_speed']} m/s
Last Updated: {weather_data['timestamp']}
----------------------------------------
"""

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_weather(city)
        
        if weather:
            print(format_weather_report(weather))
        else:
            print(f"Failed to fetch weather for {city}")
        
        time.sleep(1)
    
    print("Cache Statistics:")
    print(json.dumps(fetcher.get_cache_stats(), indent=2))
import requests
import sys

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No data to display.")
        return
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description']
    wind_speed = data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {weather_desc}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)