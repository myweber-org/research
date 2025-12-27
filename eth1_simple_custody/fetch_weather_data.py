
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        main = data['main']
        weather = data['weather'][0]
        sys = data['sys']
        
        print(f"Weather in {data['name']}, {sys['country']}:")
        print(f"  Condition: {weather['description'].title()}")
        print(f"  Temperature: {main['temp']}°C")
        print(f"  Feels like: {main['feels_like']}°C")
        print(f"  Humidity: {main['humidity']}%")
        print(f"  Pressure: {main['pressure']} hPa")
        print(f"  Wind Speed: {data['wind']['speed']} m/s")
        print(f"  Sunrise: {datetime.fromtimestamp(sys['sunrise']).strftime('%H:%M:%S')}")
        print(f"  Sunset: {datetime.fromtimestamp(sys['sunset']).strftime('%H:%M:%S')}")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to get weather data: {error_msg}")

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty")
        return
    
    weather_data = get_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()