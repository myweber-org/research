import requests
import json
from datetime import datetime

def fetch_weather_data(api_key, city):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected data structure in API response: {e}")
        return None

def display_weather_data(weather_data):
    """
    Display weather data in a formatted way.
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp']}")
    print("="*40)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    print(f"Fetching weather data for {CITY}...")
    weather_data = fetch_weather_data(API_KEY, CITY)
    
    if weather_data:
        display_weather_data(weather_data)
    else:
        print("Failed to fetch weather data.")import requests
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
            query = f"{city_name},{country_code}" if country_code else city_name
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
            'location': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'visibility': raw_data.get('visibility'),
            'clouds': raw_data.get('clouds', {}).get('all'),
            'sunrise': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunrise')).isoformat() if raw_data.get('sys', {}).get('sunrise') else None,
            'sunset': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunset')).isoformat() if raw_data.get('sys', {}).get('sunset') else None
        }
        return parsed
    
    def save_to_file(self, data, filename="weather_data.json"):
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            logging.info(f"Weather data saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save data: {e}")

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    for city in cities:
        weather_data = fetcher.get_current_weather(city)
        if weather_data:
            print(f"Weather in {city}: {weather_data['temperature']}°C, {weather_data['weather']}")
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()