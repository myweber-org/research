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
                "city": data.get("name"),
                "country": data.get("sys", {}).get("country"),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "weather": data.get("weather", [{}])[0].get("description"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "timestamp": datetime.utcfromtimestamp(data.get("dt")).isoformat(),
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
            logging.info(f"Weather data fetched for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch weather data: {e}")
            return {"error": str(e)}
        except (KeyError, ValueError) as e:
            logging.error(f"Failed to parse weather data: {e}")
            return {"error": "Invalid data format"}
    
    def save_to_file(self, data, filename="weather_data.json"):
        try:
            with open(filename, "a") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            logging.info(f"Weather data saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save data to file: {e}")
    
    def get_forecast(self, city_name, days=5, units="metric"):
        endpoint = f"{self.base_url}/forecast"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units,
            "cnt": days * 8
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch forecast: {e}")
            return {"error": str(e)}

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    for city in cities:
        weather_data = fetcher.get_current_weather(city)
        if "error" not in weather_data:
            print(f"Weather in {city}: {weather_data['temperature']}°C, {weather_data['weather']}")
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to get weather for {city}: {weather_data['error']}")

if __name__ == "__main__":
    main()