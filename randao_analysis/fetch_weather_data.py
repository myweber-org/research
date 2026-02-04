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
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
            }
            
            logging.info(f"Weather data fetched for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Data parsing error: {e}")
            return None

    def save_to_file(self, data, filename="weather_data.json"):
        if data:
            try:
                with open(filename, 'a') as f:
                    json.dump(data, f, indent=2)
                    f.write('\n')
                logging.info(f"Data saved to {filename}")
            except IOError as e:
                logging.error(f"File write error: {e}")

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    for city in cities:
        weather_data = fetcher.get_current_weather(city)
        if weather_data:
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Feels like: {weather_data['feels_like']}°C")
            print(f"  Conditions: {weather_data['weather']}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()