import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = self._process_weather_data(data)
            self.logger.info(f"Weather data fetched for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return None

    def _process_weather_data(self, raw_data):
        return {
            'city': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'visibility': raw_data.get('visibility'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'sunrise': self._convert_timestamp(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._convert_timestamp(raw_data.get('sys', {}).get('sunset')),
            'data_timestamp': self._convert_timestamp(raw_data.get('dt')),
            'timezone_offset': raw_data.get('timezone')
        }

    def _convert_timestamp(self, timestamp):
        if timestamp:
            return datetime.fromtimestamp(timestamp).isoformat()
        return None

    def save_to_file(self, data, filename="weather_data.json"):
        if data:
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                self.logger.error(f"Failed to save data to file: {e}")
                return False
        return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        weather_data = fetcher.get_weather_by_city(city)
        if weather_data:
            print(f"Weather in {city}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Conditions: {weather_data['weather_description']}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print("-" * 40)
            
            filename = f"weather_{city.lower().replace(' ', '_')}.json"
            fetcher.save_to_file(weather_data, filename)

if __name__ == "__main__":
    main()