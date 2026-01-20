import requests
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
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        main = data['main']
        weather = data['weather'][0]
        sys = data['sys']
        print(f"Weather in {data['name']}, {sys['country']}:")
        print(f"  Condition: {weather['description'].capitalize()}")
        print(f"  Temperature: {main['temp']}°C")
        print(f"  Feels like: {main['feels_like']}°C")
        print(f"  Humidity: {main['humidity']}%")
        print(f"  Pressure: {main['pressure']} hPa")
        print(f"  Wind Speed: {data['wind']['speed']} m/s")
        sunrise = datetime.fromtimestamp(sys['sunrise']).strftime('%H:%M:%S')
        sunset = datetime.fromtimestamp(sys['sunset']).strftime('%H:%M:%S')
        print(f"  Sunrise: {sunrise}")
        print(f"  Sunset: {sunset}")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to retrieve weather. Error: {error_msg}")

if __name__ == "__main__":
    API_KEY = os.environ.get('OWM_API_KEY')
    if not API_KEY:
        print("Please set the OWM_API_KEY environment variable.")
        exit(1)
    city = input("Enter city name: ").strip()
    if city:
        weather_data = get_weather(city, API_KEY)
        display_weather(weather_data)
    else:
        print("City name cannot be empty.")import requests
import json
import sys
from datetime import datetime

def fetch_weather_data(api_key, city):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def parse_weather_data(weather_json):
    """
    Parse and extract relevant information from weather JSON response.
    """
    if not weather_json or weather_json.get('cod') != 200:
        return None
    
    main_data = weather_json.get('main', {})
    weather_info = weather_json.get('weather', [{}])[0]
    
    parsed_data = {
        'city': weather_json.get('name'),
        'country': weather_json.get('sys', {}).get('country'),
        'temperature': main_data.get('temp'),
        'feels_like': main_data.get('feels_like'),
        'humidity': main_data.get('humidity'),
        'pressure': main_data.get('pressure'),
        'weather': weather_info.get('main'),
        'description': weather_info.get('description'),
        'wind_speed': weather_json.get('wind', {}).get('speed'),
        'timestamp': datetime.fromtimestamp(weather_json.get('dt')).isoformat()
    }
    
    return parsed_data

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather']} ({weather_data['description']})")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp']}")
    print("="*40)

def save_to_file(weather_data, filename='weather_data.json'):
    """
    Save weather data to a JSON file.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"Weather data saved to {filename}")
    except IOError as e:
        print(f"Error saving to file: {e}")

def main():
    """
    Main function to orchestrate weather data fetching and display.
    """
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        print("Example: python fetch_weather.py your_api_key London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather data for {city}...")
    
    raw_data = fetch_weather_data(api_key, city)
    
    if raw_data:
        parsed_data = parse_weather_data(raw_data)
        
        if parsed_data:
            display_weather(parsed_data)
            save_to_file(parsed_data)
        else:
            print("Failed to parse weather data.")
    else:
        print("Failed to fetch weather data.")

if __name__ == "__main__":
    main()