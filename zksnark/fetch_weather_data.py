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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No data to display.")
        return
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description']
    wind_speed = data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (Feels like: {feels_like}°C)")
    print(f"  Conditions: {weather_desc.capitalize()}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.cache = {}
        self.cache_duration = 300  # 5 minutes in seconds

    def get_weather(self, city_name, country_code=None):
        cache_key = f"{city_name}_{country_code}" if country_code else city_name
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                logging.info(f"Returning cached data for {city_name}")
                return cached_data
        
        # Build query parameters
        query = city_name
        if country_code:
            query += f",{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
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
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            }
            
            # Update cache
            self.cache[cache_key] = (time.time(), weather_info)
            logging.info(f"Successfully fetched weather data for {city_name}")
            
            return weather_info
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching weather for {city_name}: {e}")
            return {'error': 'Network error', 'details': str(e)}
        except KeyError as e:
            logging.error(f"Unexpected API response format for {city_name}: {e}")
            return {'error': 'Invalid API response', 'details': str(e)}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response for {city_name}: {e}")
            return {'error': 'Invalid response format', 'details': str(e)}

    def get_weather_multiple_cities(self, city_list):
        results = {}
        for city_info in city_list:
            if isinstance(city_info, dict):
                city = city_info.get('city')
                country = city_info.get('country')
                results[city] = self.get_weather(city, country)
            else:
                results[city_info] = self.get_weather(city_info)
            time.sleep(0.1)  # Small delay to avoid rate limiting
        return results

    def clear_cache(self):
        self.cache.clear()
        logging.info("Weather cache cleared")

def format_weather_output(weather_data):
    if 'error' in weather_data:
        return f"Error: {weather_data['error']} - {weather_data.get('details', 'No details')}"
    
    return f"""
Weather Report for {weather_data['city']}, {weather_data['country']}
--------------------------------------------------
Current Conditions: {weather_data['weather'].title()}
Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)
Humidity: {weather_data['humidity']}%
Pressure: {weather_data['pressure']} hPa
Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°
Visibility: {weather_data['visibility']} meters
Cloudiness: {weather_data['cloudiness']}%
Sunrise: {weather_data['sunrise']}
Sunset: {weather_data['sunset']}
Last Updated: {weather_data['timestamp']}
"""

def main():
    # Example usage - Replace with actual API key
    API_KEY = "your_api_key_here"  # Get from https://openweathermap.org/api
    
    if API_KEY == "your_api_key_here":
        print("Please replace 'your_api_key_here' with a valid OpenWeatherMap API key")
        return
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Single city example
    print("Fetching weather for London...")
    london_weather = fetcher.get_weather("London", "GB")
    print(format_weather_output(london_weather))
    
    # Multiple cities example
    print("\nFetching weather for multiple cities...")
    cities = [
        {"city": "New York", "country": "US"},
        {"city": "Tokyo", "country": "JP"},
        {"city": "Sydney", "country": "AU"}
    ]
    
    all_weather = fetcher.get_weather_multiple_cities(cities)
    
    for city, weather in all_weather.items():
        if 'error' not in weather:
            print(f"{city}: {weather['temperature']}°C, {weather['weather']}")
        else:
            print(f"{city}: Error - {weather['error']}")

if __name__ == "__main__":
    main()