
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
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch weather data: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        return {
            'city': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'visibility': raw_data.get('visibility'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'sunrise': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunrise')).isoformat(),
            'sunset': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunset')).isoformat(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_weather_forecast(self, city_name, days=5):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_forecast_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch forecast: {e}")
            return None
    
    def _parse_forecast_data(self, raw_data):
        forecast_list = []
        for item in raw_data.get('list', []):
            forecast_list.append({
                'datetime': datetime.fromtimestamp(item.get('dt')).isoformat(),
                'temperature': item.get('main', {}).get('temp'),
                'feels_like': item.get('main', {}).get('feels_like'),
                'weather': item.get('weather', [{}])[0].get('description'),
                'precipitation': item.get('pop', 0) * 100
            })
        
        return {
            'city': raw_data.get('city', {}).get('name'),
            'country': raw_data.get('city', {}).get('country'),
            'forecast': forecast_list
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    current = fetcher.get_current_weather("London", "UK")
    if current:
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Weather: {current['weather']}")
        print(f"Humidity: {current['humidity']}%")
    
    forecast = fetcher.get_weather_forecast("London", 3)
    if forecast:
        print(f"\n3-day forecast for {forecast['city']}:")
        for day in forecast['forecast'][:3]:
            print(f"{day['datetime']}: {day['temperature']}°C, {day['weather']}")

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime

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
        
        if data['cod'] != 200:
            return {'error': data.get('message', 'Unknown error')}
        
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
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {str(e)}'}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {'error': f'Data parsing error: {str(e)}'}

def display_weather(weather_data):
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*50)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)