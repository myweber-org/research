
import requests
import json
from datetime import datetime

def get_weather_data(api_key, city):
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
            return None
            
        weather_info = {
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
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def save_weather_data(data, filename='weather_data.json'):
    if data:
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            print(f"Weather data saved to {filename}")
        except IOError as e:
            print(f"Error saving data: {e}")

def display_weather_data(data):
    if data:
        print("\n" + "="*50)
        print(f"Weather Report for {data['city']}, {data['country']}")
        print("="*50)
        print(f"Temperature: {data['temperature']}°C")
        print(f"Feels like: {data['feels_like']}°C")
        print(f"Weather: {data['weather'].title()}")
        print(f"Humidity: {data['humidity']}%")
        print(f"Pressure: {data['pressure']} hPa")
        print(f"Wind: {data['wind_speed']} m/s at {data['wind_direction']}°")
        print(f"Visibility: {data['visibility']} meters")
        print(f"Cloudiness: {data['cloudiness']}%")
        print(f"Sunrise: {data['sunrise']}")
        print(f"Sunset: {data['sunset']}")
        print(f"Report Time: {data['timestamp']}")
        print("="*50)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather_data = get_weather_data(API_KEY, CITY)
    
    if weather_data:
        display_weather_data(weather_data)
        save_weather_data(weather_data)
    else:
        print("Failed to fetch weather data")import requests
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
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
        except (KeyError, json.JSONDecodeError) as e:
            return f"Error parsing weather data: {e}"

    def _parse_weather_data(self, data):
        weather_info = {
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
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    def display_weather(self, weather_data):
        if isinstance(weather_data, dict):
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
            print(f"  Conditions: {weather_data['weather'].title()}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Pressure: {weather_data['pressure']} hPa")
            print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
            print(f"  Visibility: {weather_data['visibility']} meters")
            print(f"  Cloudiness: {weather_data['cloudiness']}%")
            print(f"  Sunrise: {weather_data['sunrise']}")
            print(f"  Sunset: {weather_data['sunset']}")
            print(f"  Last updated: {weather_data['timestamp']}")
        else:
            print(weather_data)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"\n{'='*50}")
        weather = fetcher.get_weather(city)
        fetcher.display_weather(weather)

if __name__ == "__main__":
    main()import requests
import json

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        weather_info = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather_info
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected API response format: {e}")
        return None

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    if weather:
        print(json.dumps(weather, indent=2))