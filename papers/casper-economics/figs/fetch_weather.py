
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except json.JSONDecodeError:
            print("Error parsing weather response")
            return None
    
    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            print(f"API Error: {data.get('message', 'Unknown error')}")
            return None
            
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
            
        print("\n" + "="*40)
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print("="*40)
        print(f"Current Time: {weather_data['timestamp']}")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels Like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print("="*40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        
        with open(f"weather_{city_name.replace(' ', '_')}.json", 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nWeather data saved to weather_{city_name.replace(' ', '_')}.json")
    else:
        print(f"Could not retrieve weather data for {city_name}")

if __name__ == "__main__":
    main()