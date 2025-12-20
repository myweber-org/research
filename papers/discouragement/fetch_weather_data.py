
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
        """Fetch current weather data for specified location"""
        try:
            query = city_name
            if country_code:
                query += f",{country_code}"
                
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
            logging.error(f"Unexpected API response format: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        """Extract relevant information from API response"""
        return {
            'timestamp': datetime.now().isoformat(),
            'location': raw_data.get('name', 'Unknown'),
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 0),
            'visibility': raw_data.get('visibility', 0),
            'cloud_coverage': raw_data['clouds']['all']
        }
    
    def save_to_file(self, data, filename="weather_data.json"):
        """Save weather data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to save data: {e}")
            return False

def main():
    # Example usage
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Fetch weather for multiple cities
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    all_weather_data = []
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_current_weather(city)
        if weather:
            all_weather_data.append(weather)
            print(f"  Temperature: {weather['temperature']}°C")
            print(f"  Conditions: {weather['weather']}")
        else:
            print(f"  Failed to fetch data for {city}")
    
    # Save collected data
    if all_weather_data:
        fetcher.save_to_file(all_weather_data, "multi_city_weather.json")
        print("Data collection complete.")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()