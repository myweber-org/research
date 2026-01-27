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
        print(f"Error fetching weather data: {e}", file=sys.stderr)
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available.")
        return
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description.capitalize()}")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY>")
        sys.exit(1)
    api_key = sys.argv[1]
    city = sys.argv[2]
    weather = get_weather(api_key, city)
    display_weather(weather)import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        
        if data["cod"] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        main_data = data["main"]
        weather_data = data["weather"][0]
        sys_data = data["sys"]
        
        weather_info = {
            "city": data["name"],
            "country": sys_data["country"],
            "temperature": main_data["temp"],
            "feels_like": main_data["feels_like"],
            "humidity": main_data["humidity"],
            "pressure": main_data["pressure"],
            "weather": weather_data["main"],
            "description": weather_data["description"],
            "wind_speed": data["wind"]["speed"],
            "timestamp": datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        return None

def display_weather(weather_info):
    if not weather_info:
        return
        
    print("\n" + "="*40)
    print(f"Weather in {weather_info['city']}, {weather_info['country']}")
    print("="*40)
    print(f"Current Time: {weather_info['timestamp']}")
    print(f"Temperature: {weather_info['temperature']}°C")
    print(f"Feels Like: {weather_info['feels_like']}°C")
    print(f"Weather: {weather_info['weather']} - {weather_info['description']}")
    print(f"Humidity: {weather_info['humidity']}%")
    print(f"Pressure: {weather_info['pressure']} hPa")
    print(f"Wind Speed: {weather_info['wind_speed']} m/s")
    print("="*40)

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        api_key = input("Enter your OpenWeatherMap API key: ").strip()
    
    if not api_key:
        print("API key is required to fetch weather data.")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty.")
        return
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(city, api_key)
    
    if weather_data:
        display_weather(weather_data)
        
        save_option = input("\nDo you want to save this data to a file? (y/n): ").strip().lower()
        if save_option == 'y':
            filename = f"weather_{city.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            print(f"Weather data saved to {filename}")
    else:
        print("Failed to fetch weather data.")

if __name__ == "__main__":
    main()