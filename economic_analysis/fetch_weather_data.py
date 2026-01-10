
import requests
import json

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
        
        main = data["main"]
        weather_desc = data["weather"][0]["description"]
        temperature = main["temp"]
        humidity = main["humidity"]
        pressure = main["pressure"]
        
        result = {
            "city": city_name,
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "description": weather_desc
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if weather_data:
        print(f"Weather in {weather_data['city']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Conditions: {weather_data['description']}")
    else:
        print("No weather data available.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city = "London"
    
    weather = get_weather(city, API_KEY)
    display_weather(weather)
import requests
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
        wind_data = data["wind"]
        
        weather_info = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": main_data["temp"],
            "feels_like": main_data["feels_like"],
            "humidity": main_data["humidity"],
            "pressure": main_data["pressure"],
            "weather": weather_data["main"],
            "description": weather_data["description"],
            "wind_speed": wind_data["speed"],
            "wind_direction": wind_data.get("deg", "N/A"),
            "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
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
        
    print("\n" + "="*50)
    print(f"Weather in {weather_info['city']}, {weather_info['country']}")
    print("="*50)
    print(f"Temperature: {weather_info['temperature']}°C")
    print(f"Feels like: {weather_info['feels_like']}°C")
    print(f"Weather: {weather_info['weather']} ({weather_info['description']})")
    print(f"Humidity: {weather_info['humidity']}%")
    print(f"Pressure: {weather_info['pressure']} hPa")
    print(f"Wind: {weather_info['wind_speed']} m/s")
    if weather_info['wind_direction'] != "N/A":
        print(f"Wind Direction: {weather_info['wind_direction']}°")
    print(f"Last updated: {weather_info['timestamp']}")
    print("="*50)

def save_to_file(weather_info, filename="weather_data.json"):
    if not weather_info:
        return
        
    try:
        with open(filename, 'a') as f:
            json.dump(weather_info, f, indent=2)
            f.write('\n')
        print(f"Weather data saved to {filename}")
    except IOError as e:
        print(f"Error saving to file: {e}")

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set")
        print("Please set your API key: export OPENWEATHER_API_KEY='your_api_key'")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty")
        return
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(city, api_key)
    
    if weather_data:
        display_weather(weather_data)
        
        save_choice = input("\nSave this data to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_to_file(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()
import requests

def get_weather_data(api_key, city):
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
        return {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity']
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    weather = get_weather_data(API_KEY, CITY)
    if weather:
        print(f"Weather in {weather['city']}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Description: {weather['description']}")
        print(f"Humidity: {weather['humidity']}%")