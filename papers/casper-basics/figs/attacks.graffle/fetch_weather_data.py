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
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 0),
                'clouds': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
                'timestamp': datetime.now().isoformat()
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
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Data parsing error: {e}")
            return None
    
    def format_weather_report(self, weather_data: Dict[str, Any]) -> str:
        if not weather_data:
            return "No weather data available"
        
        report_lines = [
            f"Weather Report for {weather_data['city']}, {weather_data['country']}",
            f"Timestamp: {weather_data['timestamp']}",
            f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)",
            f"Weather: {weather_data['weather'].title()}",
            f"Humidity: {weather_data['humidity']}%",
            f"Pressure: {weather_data['pressure']} hPa",
            f"Wind: {weather_data['wind_speed']} m/s"
        ]
        
        if 'wind_direction' in weather_data:
            report_lines.append(f"Wind Direction: {weather_data['wind_direction']}°")
        if 'visibility' in weather_data:
            report_lines.append(f"Visibility: {weather_data['visibility']} meters")
        if 'clouds' in weather_data:
            report_lines.append(f"Cloud Coverage: {weather_data['clouds']}%")
        if 'sunrise' in weather_data and 'sunset' in weather_data:
            report_lines.append(f"Sunrise: {weather_data['sunrise']}, Sunset: {weather_data['sunset']}")
        
        return "\n".join(report_lines)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    weather_data = fetcher.get_weather_by_city("London", "UK")
    
    if weather_data:
        report = fetcher.format_weather_report(weather_data)
        print(report)
        
        with open('weather_report.txt', 'w') as f:
            f.write(report)
            print("Weather report saved to weather_report.txt")
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()