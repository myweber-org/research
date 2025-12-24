
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
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['cod'] != 200:
                return {'error': data.get('message', 'Unknown error')}
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except (KeyError, json.JSONDecodeError) as e:
            return {'error': f'Data parsing error: {str(e)}'}

def display_weather(weather_data):
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print(f"Weather in {weather_data['city']}:")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Conditions: {weather_data['description']}")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp']}")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        display_weather(weather)