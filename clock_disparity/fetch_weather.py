import requests
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
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['cod'] != 200:
                return None, f"Error: {data.get('message', 'Unknown error')}"
            
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
            return weather_info, None
            
        except requests.exceptions.RequestException as e:
            return None, f"Network error: {str(e)}"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return None, f"Data parsing error: {str(e)}"

def save_weather_data(data, filename='weather_data.json'):
    if data:
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True, None
        except IOError as e:
            return False, f"File save error: {str(e)}"
    return False, "No data to save"

def display_weather(weather_data):
    if weather_data:
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Conditions: {weather_data['description']}")
        print(f"Last updated: {weather_data['timestamp']}")
    else:
        print("No weather data available")

def main():
    api_key = "your_api_key_here"
    city = "London"
    
    fetcher = WeatherFetcher(api_key)
    weather_data, error = fetcher.get_weather(city)
    
    if error:
        print(f"Failed to fetch weather: {error}")
        return
    
    display_weather(weather_data)
    
    success, save_error = save_weather_data(weather_data)
    if success:
        print("Weather data saved successfully")
    elif save_error:
        print(f"Failed to save data: {save_error}")

if __name__ == "__main__":
    main()