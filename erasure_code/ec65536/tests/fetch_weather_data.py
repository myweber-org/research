import requests
import json
from datetime import datetime

def get_weather_data(api_key, city_name, units='metric'):
    """
    Fetch current weather data for a given city.
    
    Args:
        api_key (str): OpenWeatherMap API key
        city_name (str): Name of the city to fetch weather for
        units (str): Units of measurement ('metric', 'imperial', 'standard')
    
    Returns:
        dict: Weather data dictionary or None if request fails
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': units
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', 0),
            'cloudiness': data['clouds']['all'],
            'visibility': data.get('visibility', 0),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather_data(weather_data):
    """
    Display weather data in a formatted way.
    
    Args:
        weather_data (dict): Weather data dictionary
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Current Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels Like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather']} ({weather_data['description']})")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

def save_weather_data_to_file(weather_data, filename='weather_data.json'):
    """
    Save weather data to a JSON file.
    
    Args:
        weather_data (dict): Weather data dictionary
        filename (str): Name of the file to save data to
    """
    if not weather_data:
        print("No data to save.")
        return
    
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"Weather data saved to {filename}")
    except IOError as e:
        print(f"Error saving file: {e}")

def main():
    """
    Main function to demonstrate weather data fetching.
    """
    # Replace with your actual API key from OpenWeatherMap
    API_KEY = "your_api_key_here"
    
    if API_KEY == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key.")
        print("You can get a free API key at: https://openweathermap.org/api")
        return
    
    cities = ["London", "New York", "Tokyo", "Sydney", "Berlin"]
    
    for city in cities:
        print(f"\nFetching weather data for {city}...")
        weather_data = get_weather_data(API_KEY, city)
        
        if weather_data:
            display_weather_data(weather_data)
            save_weather_data_to_file(weather_data, f"{city.lower()}_weather.json")
        else:
            print(f"Failed to fetch weather data for {city}")
        
        # Add a small delay between requests to be respectful to the API
        import time
        time.sleep(1)

if __name__ == "__main__":
    main()