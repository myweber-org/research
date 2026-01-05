import requests
import json
import time
from datetime import datetime, timedelta
import os

class WeatherFetcher:
    CACHE_FILE = 'weather_cache.json'
    CACHE_DURATION = 300  # 5 minutes in seconds

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self._load_cache()

    def _load_cache(self):
        self.cache = {}
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except IOError:
            pass

    def _is_cache_valid(self, cache_entry):
        if not cache_entry:
            return False
        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cached_time < timedelta(seconds=self.CACHE_DURATION)

    def get_weather(self, city_name):
        cache_key = f"{city_name.lower()}"
        
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"Returning cached data for {city_name}")
            return self.cache[cache_key]['data']

        if not self.api_key:
            raise ValueError("API key not provided. Set WEATHER_API_KEY environment variable.")

        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            cache_entry = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self.cache[cache_key] = cache_entry
            self._save_cache()
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            if cache_key in self.cache:
                print("Returning stale cached data")
                return self.cache[cache_key]['data']
            raise

    def display_weather(self, city_name):
        try:
            weather_data = self.get_weather(city_name)
            
            if weather_data.get('cod') != 200:
                print(f"Error: {weather_data.get('message', 'Unknown error')}")
                return

            main = weather_data['main']
            weather = weather_data['weather'][0]
            
            print(f"Weather in {city_name}:")
            print(f"  Temperature: {main['temp']}°C")
            print(f"  Feels like: {main['feels_like']}°C")
            print(f"  Humidity: {main['humidity']}%")
            print(f"  Pressure: {main['pressure']} hPa")
            print(f"  Conditions: {weather['description'].title()}")
            print(f"  Wind Speed: {weather_data['wind']['speed']} m/s")
            
        except Exception as e:
            print(f"Failed to display weather: {e}")

def main():
    fetcher = WeatherFetcher()
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print("\n" + "="*40)
        fetcher.display_weather(city)
        time.sleep(1)

if __name__ == "__main__":
    main()