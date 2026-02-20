import requests
import json
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any

class WeatherFetcher:
    CACHE_FILE = "weather_cache.json"
    CACHE_DURATION = timedelta(minutes=30)

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self._load_cache()

    def _load_cache(self) -> None:
        if os.path.exists(self.CACHE_FILE):
            with open(self.CACHE_FILE, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def _save_cache(self) -> None:
        with open(self.CACHE_FILE, 'w') as f:
            json.dump(self.cache, f)

    def _is_cache_valid(self, city: str) -> bool:
        if city not in self.cache:
            return False
        cached_time = datetime.fromisoformat(self.cache[city]['timestamp'])
        return datetime.now() - cached_time < self.CACHE_DURATION

    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        if self._is_cache_valid(city):
            print(f"Returning cached data for {city}")
            return self.cache[city]['data']

        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            weather_info = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed']
            }

            self.cache[city] = {
                'timestamp': datetime.now().isoformat(),
                'data': weather_info
            }
            self._save_cache()

            return weather_info

        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing weather data: {e}")
            return None

    def clear_cache(self) -> None:
        self.cache = {}
        if os.path.exists(self.CACHE_FILE):
            os.remove(self.CACHE_FILE)

def main():
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        print("Please set WEATHER_API_KEY environment variable")
        return

    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}:")
        weather = fetcher.get_weather(city)
        if weather:
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Conditions: {weather['description']}")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
        else:
            print("Failed to fetch weather data")

if __name__ == "__main__":
    main()