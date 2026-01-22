import requests
import json

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
    try:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        weather_desc = data['weather'][0]['description']
        wind_speed = data['wind']['speed']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (Feels like: {feels_like}°C)")
        print(f"  Conditions: {weather_desc.capitalize()}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    except KeyError as e:
        print(f"Unexpected data structure: Missing key {e}")

if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY_HERE"
    CITY = input("Enter city name: ").strip()
    if CITY:
        weather_data = get_weather(API_KEY, CITY)
        display_weather(weather_data)
    else:
        print("City name cannot be empty.")import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import os

class WeatherFetcher:
    def __init__(self, api_key: str, cache_dir: str = "./weather_cache"):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=1)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, city: str) -> str:
        return hashlib.md5(city.lower().encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_mtime < self.cache_duration
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f)
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
                return cached['data']
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
    
    def fetch_weather(self, city: str) -> Optional[Dict[str, Any]]:
        cache_key = self._get_cache_key(city)
        
        if self._is_cache_valid(self._get_cache_path(cache_key)):
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data
        
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            processed_data = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind']['deg'],
                'timestamp': datetime.now().isoformat(),
                'source': 'api'
            }
            
            self._save_to_cache(cache_key, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error parsing weather data: {e}")
            return None
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        if not weather_data:
            print("No weather data available")
            return
        
        print(f"Weather for {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Conditions: {weather_data['weather']} ({weather_data['description']})")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
        print(f"  Source: {weather_data['source']}")
        print(f"  Last updated: {weather_data['timestamp']}")

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print("\n" + "="*50)
        weather_data = fetcher.fetch_weather(city)
        fetcher.display_weather(weather_data)
        time.sleep(1)

if __name__ == "__main__":
    main()