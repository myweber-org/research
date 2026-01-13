import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
        except json.JSONDecodeError:
            return "Error parsing weather data"

    def _parse_weather_data(self, data):
        weather_info = {
            'city': data.get('name'),
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    def display_weather(self, city_name):
        weather = self.get_weather(city_name)
        if isinstance(weather, dict):
            print(f"Weather in {weather['city']} at {weather['timestamp']}:")
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Feels like: {weather['feels_like']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Pressure: {weather['pressure']} hPa")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            print(f"Conditions: {weather['description']}")
        else:
            print(weather)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        fetcher.display_weather(city)
        print("-" * 40)

if __name__ == "__main__":
    main()