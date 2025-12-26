
import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_current_weather(self, city_name, units="metric"):
        endpoint = f"{self.base_url}/weather"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"]["deg"],
                "visibility": data.get("visibility", "N/A"),
                "cloudiness": data["clouds"]["all"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
            }
            
            self.logger.info(f"Weather data fetched for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching weather data: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Data parsing error: {e}")
            return None

    def save_to_json(self, data, filename="weather_data.json"):
        if data:
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                self.logger.info(f"Data saved to {filename}")
                return True
            except IOError as e:
                self.logger.error(f"File save error: {e}")
                return False
        return False

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    all_weather_data = {}
    for city in cities:
        weather_data = fetcher.get_current_weather(city)
        if weather_data:
            all_weather_data[city] = weather_data
            print(f"Current weather in {city}: {weather_data['temperature']}°C, {weather_data['weather']}")
        else:
            print(f"Failed to fetch weather data for {city}")
    
    if all_weather_data:
        fetcher.save_to_json(all_weather_data, "multi_city_weather.json")

if __name__ == "__main__":
    main()