import requests
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
        
        if data['cod'] != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        
        weather_info = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"Data parsing error: {str(e)}"

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    result = get_weather(API_KEY, CITY)
    if isinstance(result, dict):
        print(f"Weather in {result['city']}:")
        print(f"Temperature: {result['temperature']}°C")
        print(f"Conditions: {result['description']}")
        print(f"Humidity: {result['humidity']}%")
        print(f"Wind Speed: {result['wind_speed']} m/s")
    else:
        print(result)