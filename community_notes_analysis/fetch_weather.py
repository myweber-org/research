import requests
import os

def get_weather(city_name):
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return "API key not found. Set OPENWEATHER_API_KEY environment variable."
    
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
            return f"Error: {data.get('message', 'Unknown error')}"
        
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        
        return f"Weather in {city_name}: {weather_desc}, Temperature: {temp}°C, Humidity: {humidity}%"
    
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Data parsing error: {str(e)}"

if __name__ == "__main__":
    city = input("Enter city name: ")
    print(get_weather(city))