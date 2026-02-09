import requests
import json
from datetime import datetime

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
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None

def save_weather_data(data, filename='weather_data.json'):
    if data:
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Weather data saved to {filename}")
        except IOError as e:
            print(f"Error saving data: {e}")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather(city)
        
        if weather_data:
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Conditions: {weather_data['description']}")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            
            filename = f"{city.lower().replace(' ', '_')}_weather.json"
            save_weather_data(weather_data, filename)import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_current_weather(self, city_name):
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
            logging.info(f"Weather data fetched for {city_name}")
            return weather_info
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Data parsing error: {e}")
            return None

    def save_to_file(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, 'a') as f:
                    json.dump(weather_data, f, indent=2)
                    f.write('\n')
                logging.info(f"Weather data saved to {filename}")
            except IOError as e:
                logging.error(f"File write error: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        weather = fetcher.get_current_weather(city)
        if weather:
            print(f"Weather in {weather['city']}: {weather['temperature']}°C, {weather['description']}")
            fetcher.save_to_file(weather)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()