
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch weather data from OpenWeatherMap API"""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch weather data for a given city
        
        Args:
            city: Name of the city
            country_code: Optional country code (e.g., 'US', 'GB')
        
        Returns:
            Dictionary containing weather data
        
        Raises:
            ValueError: If city is empty
            ConnectionError: If API request fails
        """
        if not city or not city.strip():
            raise ValueError("City name cannot be empty")
        
        query = city.strip()
        if country_code:
            query = f"{query},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather data: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid response from API: {str(e)}")
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure raw API response"""
        main = raw_data.get('main', {})
        weather = raw_data.get('weather', [{}])[0]
        wind = raw_data.get('wind', {})
        
        return {
            'city': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', 'Unknown'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'weather_condition': weather.get('description', 'Unknown'),
            'weather_icon': weather.get('icon'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'timestamp': datetime.utcnow().isoformat(),
            'api_timestamp': raw_data.get('dt')
        }
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        """Display weather information in a readable format"""
        print("\n" + "="*50)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Condition: {weather_data['weather_condition'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s")
        print(f"Last updated: {weather_data['timestamp']}")
        print("="*50 + "\n")

def main():
    """Example usage of the WeatherFetcher class"""
    # Note: In production, use environment variables for API key
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Example cities to fetch weather for
    cities = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        try:
            print(f"\nFetching weather for {city}, {country}...")
            weather_data = fetcher.get_weather(city, country)
            fetcher.display_weather(weather_data)
        except Exception as e:
            print(f"Error fetching weather for {city}: {e}")

if __name__ == "__main__":
    main()