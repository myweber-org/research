import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_current_weather(self, city_name):
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                self.logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return None

    def _parse_weather_data(self, data):
        parsed_data = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().isoformat()
        }
        return parsed_data

    def save_to_file(self, data, filename="weather_data.json"):
        if not data:
            self.logger.warning("No data to save")
            return False
        
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            self.logger.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            self.logger.error(f"Failed to save data: {e}")
            return False

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Sydney", "Berlin"]
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather_data = fetcher.get_current_weather(city)
        
        if weather_data:
            print(f"Temperature in {weather_data['city']}: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather']} ({weather_data['description']})")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()