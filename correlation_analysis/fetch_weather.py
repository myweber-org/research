
import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather_by_city(self, city_name, units='metric'):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._format_weather_data(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
        except json.JSONDecodeError:
            return "Error parsing weather data"

    def get_weather_by_coords(self, lat, lon, units='metric'):
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._format_weather_data(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
        except json.JSONDecodeError:
            return "Error parsing weather data"

    def _format_weather_data(self, data):
        if data.get('cod') != 200:
            return f"API Error: {data.get('message', 'Unknown error')}"

        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        sys = data.get('sys', {})

        formatted = {
            'location': f"{data.get('name')}, {sys.get('country')}",
            'coordinates': f"{data.get('coord', {}).get('lat')}, {data.get('coord', {}).get('lon')}",
            'temperature': f"{main.get('temp')}°C",
            'feels_like': f"{main.get('feels_like')}°C",
            'humidity': f"{main.get('humidity')}%",
            'pressure': f"{main.get('pressure')} hPa",
            'weather': weather.get('description', '').title(),
            'wind_speed': f"{wind.get('speed')} m/s",
            'wind_direction': wind.get('deg'),
            'cloudiness': f"{data.get('clouds', {}).get('all')}%",
            'visibility': f"{data.get('visibility', 0) / 1000} km" if data.get('visibility') else 'N/A',
            'sunrise': datetime.fromtimestamp(sys.get('sunrise')).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(sys.get('sunset')).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data.get('dt')).strftime('%Y-%m-%d %H:%M:%S')
        }
        return formatted

    def print_weather(self, weather_data):
        if isinstance(weather_data, dict):
            print("\n" + "="*50)
            print(f"Weather Report for {weather_data['location']}")
            print("="*50)
            for key, value in weather_data.items():
                if key != 'location':
                    print(f"{key.replace('_', ' ').title():20}: {value}")
            print("="*50)
        else:
            print(weather_data)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Or: python fetch_weather.py <lat> <lon>")
        sys.exit(1)

    fetcher = WeatherFetcher()

    if len(sys.argv) == 2:
        city = sys.argv[1]
        weather = fetcher.get_weather_by_city(city)
    else:
        try:
            lat = float(sys.argv[1])
            lon = float(sys.argv[2])
            weather = fetcher.get_weather_by_coords(lat, lon)
        except ValueError:
            print("Error: Latitude and longitude must be numeric values")
            sys.exit(1)

    fetcher.print_weather(weather)

if __name__ == "__main__":
    main()