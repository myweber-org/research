
import requests
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query = f"{city},{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = self._process_weather_data(data)
            logger.info(f"Weather data fetched for {query}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching weather: {e}")
            return {'error': str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return {'error': 'Invalid response from server'}
        except KeyError as e:
            logger.error(f"Missing expected data in response: {e}")
            return {'error': 'Unexpected response format'}
            
    def _process_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'city': raw_data.get('name', 'Unknown'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'timestamp': datetime.now().isoformat()
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    for city, country in cities:
        print(f"\nFetching weather for {city}, {country}:")
        weather = fetcher.get_weather(city, country)
        
        if 'error' not in weather:
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Weather: {weather['weather']}")
            print(f"Humidity: {weather['humidity']}%")
        else:
            print(f"Error: {weather['error']}")

if __name__ == "__main__":
    main()