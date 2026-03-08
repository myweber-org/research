
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query = f"{city},{country_code}"
        
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
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 'N/A'),
                'visibility': data.get('visibility', 'N/A'),
                'cloudiness': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {'error': f"Network error: {str(e)}"}
        except KeyError as e:
            return {'error': f"Invalid response format: missing key {str(e)}"}
        except json.JSONDecodeError:
            return {'error': "Invalid JSON response"}
    
    def format_weather_report(self, weather_data: Dict[str, Any]) -> str:
        if 'error' in weather_data:
            return f"Error: {weather_data['error']}"
        
        report_lines = [
            f"Weather Report for {weather_data['city']}, {weather_data['country']}",
            f"Timestamp: {weather_data['timestamp']}",
            "=" * 50,
            f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)",
            f"Conditions: {weather_data['description'].title()}",
            f"Humidity: {weather_data['humidity']}%",
            f"Pressure: {weather_data['pressure']} hPa",
            f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°",
            f"Visibility: {weather_data['visibility']} meters",
            f"Cloudiness: {weather_data['cloudiness']}%",
            f"Sunrise: {weather_data['sunrise']}",
            f"Sunset: {weather_data['sunset']}"
        ]
        
        return "\n".join(report_lines)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Paris", "FR")
    ]
    
    for city, country in cities:
        print(f"\nFetching weather for {city}, {country}...")
        weather = fetcher.get_weather(city, country)
        report = fetcher.format_weather_report(weather)
        print(report)

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_weather_by_coords(self, lat, lon, units="metric"):
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def _parse_weather_data(self, data):
        if not data or "main" not in data:
            return None
        
        weather_info = {
            "city": data.get("name", "Unknown"),
            "country": data.get("sys", {}).get("country", "Unknown"),
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather": data["weather"][0]["description"] if data.get("weather") else "Unknown",
            "wind_speed": data.get("wind", {}).get("speed", 0),
            "wind_direction": data.get("wind", {}).get("deg", 0),
            "visibility": data.get("visibility", 0),
            "cloudiness": data.get("clouds", {}).get("all", 0),
            "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M:%S") if "sys" in data else "Unknown",
            "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M:%S") if "sys" in data else "Unknown",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return weather_info
    
    def save_to_file(self, weather_data, filename="weather_data.json"):
        if not weather_data:
            print("No weather data to save")
            return False
        
        try:
            with open(filename, "a") as f:
                json.dump(weather_data, f, indent=2)
                f.write("\n")
            print(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            print(f"Error saving to file: {e}")
            return False
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data to display")
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
        print(f"Report Time: {weather_data['timestamp']}")
        print("="*50 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city_name = " ".join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get a free API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    fetcher = WeatherFetcher(api_key)
    
    print(f"Fetching weather data for {city_name}...")
    weather_data = fetcher.get_weather_by_city(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        fetcher.save_to_file(weather_data)
    else:
        print(f"Could not fetch weather data for {city_name}")

if __name__ == "__main__":
    main()import requests
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
        print("No weather data to display.")
        return
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']
    wind_speed = data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C")
    print(f"  Humidity: {humidity}%")
    print(f"  Conditions: {description}")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()