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
import requests
import json
import sys
from datetime import datetime

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
    
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        wind_speed = weather_data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C (Feels like: {feels_like}°C)")
        print(f"Conditions: {description.capitalize()}")
        print(f"Humidity: {humidity}%")
        print(f"Wind Speed: {wind_speed} m/s")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyError as e:
        print(f"Unexpected data format: Missing key {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get a free API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    print(f"Fetching weather for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data and weather_data.get('cod') == 200:
        display_weather(weather_data)
    else:
        error_msg = weather_data.get('message', 'Unknown error') if weather_data else 'Connection failed'
        print(f"Failed to get weather data: {error_msg}")

if __name__ == "__main__":
    main()