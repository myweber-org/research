
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
            logging.error(f"Network error fetching weather: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response: {e}")
            return None
        except KeyError as e:
            logging.error(f"Unexpected data structure: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        parsed = {
            'timestamp': datetime.now().isoformat(),
            'location': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', ''),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'visibility': raw_data.get('visibility'),
            'sunrise': raw_data.get('sys', {}).get('sunrise'),
            'sunset': raw_data.get('sys', {}).get('sunset')
        }
        return parsed
    
    def save_to_file(self, data, filename="weather_data.json"):
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            logging.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to save data: {e}")
            return False

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        print(f"Fetching weather for {city}, {country}...")
        weather_data = fetcher.get_current_weather(city, country)
        
        if weather_data:
            print(f"Temperature in {weather_data['location']}: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print("-" * 40)
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()