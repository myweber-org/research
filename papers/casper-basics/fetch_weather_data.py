import requests
import json

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        weather_info = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = "London"
    
    weather = get_weather(API_KEY, city_name)
    if weather:
        print(f"Weather in {weather['city']}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Conditions: {weather['description']}")
        print(f"Wind Speed: {weather['wind_speed']} m/s")
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()
        
    def get_weather(self, city_name, units='metric'):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None
    
    def _parse_response(self, data):
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
    
    def save_to_file(self, weather_data, filename='weather_data.json'):
        if weather_data:
            try:
                with open(filename, 'a') as f:
                    json.dump(weather_data, f, indent=2)
                    f.write('\n')
                print(f"Weather data saved to {filename}")
            except IOError as e:
                print(f"Error saving to file: {e}")
    
    def display_weather(self, weather_data):
        if weather_data:
            print("\n" + "="*50)
            print(f"Weather in {weather_data['city']}, {weather_data['country']}")
            print("="*50)
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Feels like: {weather_data['feels_like']}°C")
            print(f"Weather: {weather_data['weather'].title()}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Pressure: {weather_data['pressure']} hPa")
            print(f"Wind: {weather_data['wind_speed']} m/s")
            print(f"Visibility: {weather_data['visibility']} meters")
            print(f"Cloudiness: {weather_data['cloudiness']}%")
            print(f"Sunrise: {weather_data['sunrise']}")
            print(f"Sunset: {weather_data['sunset']}")
            print(f"Last updated: {weather_data['timestamp']}")
            print("="*50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    fetcher = WeatherFetcher(api_key)
    
    weather_data = fetcher.get_weather(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        fetcher.save_to_file(weather_data)
    else:
        print("Failed to retrieve weather data.")

if __name__ == "__main__":
    main()