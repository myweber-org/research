
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