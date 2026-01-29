
import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        
    def get_current_weather(self, city_name, country_code=None):
        try:
            query = city_name
            if country_code:
                query = f"{city_name},{country_code}"
                
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        parsed = {
            'timestamp': datetime.now().isoformat(),
            'location': {
                'city': raw_data.get('name'),
                'country': raw_data.get('sys', {}).get('country'),
                'coordinates': raw_data.get('coord')
            },
            'temperature': {
                'current': raw_data.get('main', {}).get('temp'),
                'feels_like': raw_data.get('main', {}).get('feels_like'),
                'min': raw_data.get('main', {}).get('temp_min'),
                'max': raw_data.get('main', {}).get('temp_max')
            },
            'conditions': {
                'main': raw_data.get('weather', [{}])[0].get('main'),
                'description': raw_data.get('weather', [{}])[0].get('description'),
                'icon': raw_data.get('weather', [{}])[0].get('icon')
            },
            'wind': {
                'speed': raw_data.get('wind', {}).get('speed'),
                'direction': raw_data.get('wind', {}).get('deg')
            },
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'visibility': raw_data.get('visibility'),
            'clouds': raw_data.get('clouds', {}).get('all')
        }
        return parsed
    
    def save_to_file(self, data, filename="weather_data.json"):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to save data: {e}")
            return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    weather_data = fetcher.get_current_weather("London", "UK")
    
    if weather_data:
        print(json.dumps(weather_data, indent=2))
        fetcher.save_to_file(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()