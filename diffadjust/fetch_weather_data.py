import requests
import sys
import os

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
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    print(f"\nWeather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Conditions: {weather_data['weather'].title()}")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        print("Example: python fetch_weather_data.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set")
        print("Please set your OpenWeatherMap API key as environment variable")
        sys.exit(1)
    
    weather_data = get_weather(city_name, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
import os

def get_current_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        if weather_data.get('cod') != 200:
            error_message = weather_data.get('message', 'Unknown error')
            raise Exception(f"API error: {error_message}")

        return {
            'city': weather_data['name'],
            'country': weather_data['sys']['country'],
            'temperature': weather_data['main']['temp'],
            'feels_like': weather_data['main']['feels_like'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'weather': weather_data['weather'][0]['main'],
            'description': weather_data['weather'][0]['description'],
            'wind_speed': weather_data['wind']['speed'],
            'wind_deg': weather_data['wind']['deg'],
            'visibility': weather_data.get('visibility', 'N/A'),
            'clouds': weather_data['clouds']['all']
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {e}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {e}")

def display_weather(weather_info):
    """
    Display weather information in a readable format.
    """
    print(f"Weather in {weather_info['city']}, {weather_info['country']}:")
    print(f"  Temperature: {weather_info['temperature']}°C (feels like {weather_info['feels_like']}°C)")
    print(f"  Conditions: {weather_info['weather']} - {weather_info['description']}")
    print(f"  Humidity: {weather_info['humidity']}%")
    print(f"  Pressure: {weather_info['pressure']} hPa")
    print(f"  Wind: {weather_info['wind_speed']} m/s at {weather_info['wind_deg']}°")
    print(f"  Cloudiness: {weather_info['clouds']}%")
    if weather_info['visibility'] != 'N/A':
        print(f"  Visibility: {weather_info['visibility']} meters")

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty.")
        else:
            weather = get_current_weather(city)
            display_weather(weather)
    except Exception as e:
        print(f"Error: {e}")