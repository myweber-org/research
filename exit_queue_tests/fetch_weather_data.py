
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
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available.")
        return
        
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print("="*40)

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py your_api_key London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def get_weather_by_city(self, city_name: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = city_name
        if country_code:
            query += f",{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected data structure: {e}")
            return None

    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        parsed_data = {
            'city': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', 'Unknown'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description', 'Unknown'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'visibility': raw_data.get('visibility'),
            'sunrise': self._convert_timestamp(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._convert_timestamp(raw_data.get('sys', {}).get('sunset')),
            'data_timestamp': self._convert_timestamp(raw_data.get('dt')),
            'timezone_offset': raw_data.get('timezone')
        }
        return parsed_data

    def _convert_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        if timestamp is None:
            return None
        return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        if not weather_data:
            print("No weather data available.")
            return
        
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Conditions: {weather_data['weather_description'].title()}")
        print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"  Cloudiness: {weather_data['cloudiness']}%")
        print(f"  Visibility: {weather_data['visibility']} meters")
        print(f"  Sunrise: {weather_data['sunrise']}")
        print(f"  Sunset: {weather_data['sunset']}")
        print(f"  Data collected at: {weather_data['data_timestamp']}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities_to_check = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities_to_check:
        print(f"\n{'='*50}")
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_weather_by_city(city)
        fetcher.display_weather(weather)

if __name__ == "__main__":
    main()