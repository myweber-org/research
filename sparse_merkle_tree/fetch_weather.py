
import requests
import json
import time
from datetime import datetime, timedelta

class WeatherFetcher:
    def __init__(self, api_key, cache_duration=300):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache = {}
        self.cache_duration = cache_duration

    def get_weather(self, city):
        cache_key = city.lower()
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                print(f"Returning cached data for {city}")
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
            
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache[cache_key] = (weather_info, time.time())
            return weather_info
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather for {city}: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing weather data for {city}: {e}")
            return None

    def clear_cache(self):
        self.cache.clear()
        print("Cache cleared")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        
        if weather:
            print(f"City: {weather['city']}")
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Conditions: {weather['description']}")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            print(f"Last Updated: {weather['timestamp']}")
        else:
            print(f"Failed to fetch weather for {city}")
    
    print(f"\nCache size: {len(fetcher.cache)}")
    fetcher.clear_cache()

if __name__ == "__main__":
    main()