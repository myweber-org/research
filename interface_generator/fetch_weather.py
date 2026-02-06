
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