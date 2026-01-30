import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WeatherFetcher/1.0"})

    def get_current_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        location = f"{city},{country_code}" if country_code else city
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "location": data.get("name", "Unknown"),
                "country": data.get("sys", {}).get("country", "Unknown"),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "weather": data.get("weather", [{}])[0].get("description"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "wind_direction": data.get("wind", {}).get("deg"),
                "visibility": data.get("visibility"),
                "cloudiness": data.get("clouds", {}).get("all"),
                "sunrise": datetime.fromtimestamp(data.get("sys", {}).get("sunrise", 0)).isoformat() if data.get("sys", {}).get("sunrise") else None,
                "sunset": datetime.fromtimestamp(data.get("sys", {}).get("sunset", 0)).isoformat() if data.get("sys", {}).get("sunset") else None
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}", "timestamp": datetime.utcnow().isoformat()}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}", "timestamp": datetime.utcnow().isoformat()}
        except KeyError as e:
            return {"error": f"Missing expected data in response: {str(e)}", "timestamp": datetime.utcnow().isoformat()}

    def get_weather_forecast(self, city: str, days: int = 5) -> Dict[str, Any]:
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
            "cnt": days * 8
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data.get("list", [])[:days*8:8]:
                forecast = {
                    "datetime": datetime.fromtimestamp(item.get("dt", 0)).isoformat(),
                    "temperature": item.get("main", {}).get("temp"),
                    "feels_like": item.get("main", {}).get("feels_like"),
                    "humidity": item.get("main", {}).get("humidity"),
                    "weather": item.get("weather", [{}])[0].get("description"),
                    "wind_speed": item.get("wind", {}).get("speed"),
                    "precipitation": item.get("pop", 0) * 100
                }
                forecasts.append(forecast)
            
            return {
                "city": data.get("city", {}).get("name"),
                "country": data.get("city", {}).get("country"),
                "forecast_days": days,
                "forecasts": forecasts,
                "fetched_at": datetime.utcnow().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}", "timestamp": datetime.utcnow().isoformat()}

def save_weather_data(data: Dict[str, Any], filename: str = "weather_data.json") -> bool:
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except (IOError, TypeError) as e:
        print(f"Failed to save data: {e}")
        return False

def display_weather_summary(data: Dict[str, Any]) -> None:
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    
    print(f"Weather in {data.get('location', 'Unknown')}, {data.get('country', 'Unknown')}")
    print(f"Temperature: {data.get('temperature')}°C (feels like {data.get('feels_like')}°C)")
    print(f"Conditions: {data.get('weather', 'Unknown')}")
    print(f"Humidity: {data.get('humidity')}%")
    print(f"Wind: {data.get('wind_speed')} m/s at {data.get('wind_direction')}°")
    print(f"Updated: {data.get('timestamp', 'Unknown')}")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    current_weather = fetcher.get_current_weather("London", "GB")
    display_weather_summary(current_weather)
    
    if "error" not in current_weather:
        save_weather_data(current_weather, "london_weather.json")
    
    forecast = fetcher.get_weather_forecast("London", 3)
    if "error" not in forecast:
        print(f"\n3-day forecast for {forecast.get('city')}:")
        for day in forecast.get("forecasts", []):
            print(f"{day['datetime']}: {day['temperature']}°C, {day['weather']}")