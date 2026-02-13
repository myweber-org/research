import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def get_weather(self, city_name, country_code=''):
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENWEATHER_API_KEY environment variable.")
        
        query = f"{city_name},{country_code}" if country_code else city_name
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            return {'error': f"Failed to fetch weather data: {str(e)}"}
        except json.JSONDecodeError:
            return {'error': "Invalid response from weather API"}
    
    def _parse_response(self, data):
        if data.get('cod') != 200:
            return {'error': data.get('message', 'Unknown error')}
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 0),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_to_file(self, data, filename='weather_data.json'):
        if 'error' not in data:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            return True
        return False

def main():
    fetcher = WeatherFetcher()
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather_data = fetcher.get_weather(city)
        
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
        else:
            print(f"Temperature in {weather_data['city']}: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind: {weather_data['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather_data)

if __name__ == "__main__":
    main()