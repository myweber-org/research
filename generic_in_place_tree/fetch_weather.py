import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather_by_city(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query += f",{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to fetch weather data: {str(e)}'}
        except json.JSONDecodeError:
            return {'error': 'Invalid response from weather service'}
    
    def get_weather_by_coords(self, lat, lon):
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to fetch weather data: {str(e)}'}
    
    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            return {'error': data.get('message', 'Unknown error')}
        
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        sys = data.get('sys', {})
        
        return {
            'location': data.get('name'),
            'country': sys.get('country'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'weather': weather.get('main'),
            'description': weather.get('description'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'visibility': data.get('visibility'),
            'cloudiness': data.get('clouds', {}).get('all'),
            'sunrise': datetime.fromtimestamp(sys.get('sunrise')).strftime('%H:%M:%S') if sys.get('sunrise') else None,
            'sunset': datetime.fromtimestamp(sys.get('sunset')).strftime('%H:%M:%S') if sys.get('sunset') else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def display_weather(self, weather_data):
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
        
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['location']}, {weather_data['country']}")
        print("="*50)
        print(f"Current Time: {weather_data['timestamp']}")
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print("="*50)

def main():
    api_key = "your_api_key_here"  # Replace with actual API key or set environment variable
    
    fetcher = WeatherFetcher(api_key)
    
    # Example: Get weather by city
    print("Fetching weather for London, UK...")
    weather = fetcher.get_weather_by_city("London", "UK")
    fetcher.display_weather(weather)
    
    # Example: Get weather by coordinates (New York)
    print("\nFetching weather for New York by coordinates...")
    weather = fetcher.get_weather_by_coords(40.7128, -74.0060)
    fetcher.display_weather(weather)

if __name__ == "__main__":
    main()