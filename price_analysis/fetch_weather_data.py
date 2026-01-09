import requests
import json
import sys

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
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except json.JSONDecodeError:
        print("Error parsing response")
        return None

def display_weather(data):
    if not data:
        return
        
    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']
    wind_speed = data['wind']['speed']
    
    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {description}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def get_weather_by_city(self, city_name: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind']['deg'],
                'visibility': data.get('visibility', 'N/A'),
                'cloudiness': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Data parsing error: {e}")
            return None

    def get_weather_by_coordinates(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'weather': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None

def display_weather_data(weather_data: Dict[str, Any]) -> None:
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*50)
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
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

def main():
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    print("Weather Data Fetcher")
    print("1. Get weather by city name")
    print("2. Get weather by coordinates")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        city = input("Enter city name: ").strip()
        country = input("Enter country code (optional, press Enter to skip): ").strip()
        country = country if country else None
        
        weather_data = fetcher.get_weather_by_city(city, country)
        display_weather_data(weather_data)
        
    elif choice == '2':
        try:
            lat = float(input("Enter latitude: ").strip())
            lon = float(input("Enter longitude: ").strip())
            
            weather_data = fetcher.get_weather_by_coordinates(lat, lon)
            display_weather_data(weather_data)
            
        except ValueError:
            print("Invalid coordinates. Please enter numeric values.")
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()