import requests
import json
import sys
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city):
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
        except json.JSONDecodeError:
            return "Error parsing weather data"
    
    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            return f"API Error: {data.get('message', 'Unknown error')}"
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
    
    def display_weather(self, weather_data):
        if isinstance(weather_data, dict):
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Feels like: {weather_data['feels_like']}°C")
            print(f"  Conditions: {weather_data['weather'].title()}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Pressure: {weather_data['pressure']} hPa")
            print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
            print(f"  Last Updated: {weather_data['timestamp']}")
        else:
            print(weather_data)

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city)
    fetcher.display_weather(weather_data)

if __name__ == "__main__":
    main()