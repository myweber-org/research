
import requests
import json
import time
from datetime import datetime, timedelta
import os

class WeatherFetcher:
    def __init__(self, api_key, cache_dir='./weather_cache'):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(minutes=30)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, city):
        safe_name = city.lower().replace(' ', '_')
        return os.path.join(self.cache_dir, f"{safe_name}.json")

    def _is_cache_valid(self, cache_path):
        if not os.path.exists(cache_path):
            return False
        
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - mod_time < self.cache_duration

    def fetch_weather(self, city):
        cache_path = self._get_cache_path(city)
        
        if self._is_cache_valid(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return None

    def display_weather(self, city):
        data = self.fetch_weather(city)
        
        if not data or data.get('cod') != 200:
            print(f"Could not retrieve weather for {city}")
            return
        
        main = data['main']
        weather = data['weather'][0]
        
        print(f"Weather in {city}:")
        print(f"  Temperature: {main['temp']}°C")
        print(f"  Feels like: {main['feels_like']}°C")
        print(f"  Conditions: {weather['description']}")
        print(f"  Humidity: {main['humidity']}%")
        print(f"  Pressure: {main['pressure']} hPa")

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        fetcher.display_weather(city)
        print()

if __name__ == "__main__":
    main()
import requests
import json
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()