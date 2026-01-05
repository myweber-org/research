
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query += f",{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'visibility': data.get('visibility', 'N/A'),
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except (KeyError, json.JSONDecodeError) as e:
            return {'error': f'Data parsing error: {str(e)}'}
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
            
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Conditions: {weather_data['weather'].title()}")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
        if weather_data['visibility'] != 'N/A':
            print(f"  Visibility: {weather_data['visibility']} meters")
        print(f"  Last updated: {weather_data['timestamp']}")

def main():
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities_to_check = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Paris", "FR")
    ]
    
    for city, country in cities_to_check:
        print("\n" + "="*50)
        weather = fetcher.get_weather(city, country)
        fetcher.display_weather(weather)

if __name__ == "__main__":
    main()