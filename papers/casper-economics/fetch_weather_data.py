
import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def get_current_weather(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"

        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None

    def _parse_weather_data(self, data):
        if not data or 'main' not in data:
            return None

        return {
            'city': data.get('name', 'Unknown'),
            'temperature': data['main'].get('temp'),
            'humidity': data['main'].get('humidity'),
            'pressure': data['main'].get('pressure'),
            'description': data['weather'][0].get('description') if data.get('weather') else 'N/A',
            'wind_speed': data['wind'].get('speed') if data.get('wind') else 0,
            'timestamp': datetime.fromtimestamp(data.get('dt', 0)).isoformat() if data.get('dt') else None
        }

    def get_weather_forecast(self, city_name, days=5):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days
        }

        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return self._parse_forecast_data(response.json())
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch forecast: {e}")
            return None

    def _parse_forecast_data(self, data):
        if not data or 'list' not in data:
            return None

        forecasts = []
        for item in data['list']:
            forecast = {
                'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'description': item['weather'][0]['description'] if item.get('weather') else 'N/A'
            }
            forecasts.append(forecast)

        return {
            'city': data['city']['name'],
            'country': data['city']['country'],
            'forecasts': forecasts
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)

    current = fetcher.get_current_weather("London", "UK")
    if current:
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Humidity: {current['humidity']}%")
        print(f"Conditions: {current['description']}")

    forecast = fetcher.get_weather_forecast("Tokyo", 3)
    if forecast:
        print(f"\nForecast for {forecast['city']}, {forecast['country']}:")
        for day in forecast['forecasts']:
            print(f"{day['datetime']}: {day['temperature']}°C, {day['description']}")

if __name__ == "__main__":
    main()import requests
import json

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
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description}")
    else:
        print("City not found or invalid data.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_data = get_weather(API_KEY, city_name)
    display_weather(weather_data)