
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
    main()