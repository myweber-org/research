
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

class WeatherFetcher:
    CACHE_DIR = Path("./weather_cache")
    CACHE_DURATION = timedelta(minutes=30)
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def _get_cache_path(self, city: str) -> Path:
        return self.CACHE_DIR / f"{city.lower().replace(' ', '_')}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.CACHE_DURATION
    
    def _read_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _write_cache(self, cache_path: Path, data: Dict[str, Any]) -> None:
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass
    
    def fetch_weather(self, city: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(city)
        
        if self._is_cache_valid(cache_path):
            cached_data = self._read_cache(cache_path)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                return None
            
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
                'timestamp': datetime.now().isoformat(),
                'source': 'api'
            }
            
            self._write_cache(cache_path, processed_data)
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
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"  Source: {weather_data['source']}")
        print(f"  Last updated: {weather_data['timestamp']}")

def main():
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\n{'='*40}")
        weather_data = fetcher.fetch_weather(city)
        fetcher.display_weather(weather_data)
        time.sleep(1)

if __name__ == "__main__":
    main()