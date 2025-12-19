
import requests
import json
import sys

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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
    
    if weather_data.get('cod') != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
        return
    
    main = weather_data['main']
    weather = weather_data['weather'][0]
    
    print(f"Weather in {weather_data['name']}, {weather_data['sys']['country']}:")
    print(f"  Temperature: {main['temp']}°C")
    print(f"  Feels like: {main['feels_like']}°C")
    print(f"  Conditions: {weather['description'].capitalize()}")
    print(f"  Humidity: {main['humidity']}%")
    print(f"  Pressure: {main['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind']['speed']} m/s")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        print("Please set your OpenWeatherMap API key as WEATHER_API_KEY environment variable")
        sys.exit(1)
    
    city = ' '.join(sys.argv[1:])
    api_key = "YOUR_API_KEY_HERE"
    
    # In production, get API key from environment variable
    # import os
    # api_key = os.getenv('WEATHER_API_KEY', 'YOUR_API_KEY_HERE')
    
    if api_key == "YOUR_API_KEY_HERE":
        print("Error: Please replace 'YOUR_API_KEY_HERE' with your actual OpenWeatherMap API key")
        print("Get a free API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()