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
            
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return weather_info
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None

    def display_weather(self, city_name):
        weather_data = self.get_weather(city_name)
        
        if weather_data:
            print(f"\nWeather in {weather_data['city']}:")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Feels like: {weather_data['feels_like']}°C")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Pressure: {weather_data['pressure']} hPa")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            print(f"Conditions: {weather_data['description']}")
            print(f"Last updated: {weather_data['timestamp']}")
        else:
            print(f"Could not retrieve weather data for {city_name}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        fetcher.display_weather(city)
        print("-" * 40)

if __name__ == "__main__":
    main()