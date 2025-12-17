import requests
import json
import os

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    
    response = requests.get(complete_url)
    data = response.json()
    
    if data["cod"] != "404":
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
    else:
        return {"error": "City not found"}

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set.")
        return
    
    city = input("Enter city name: ")
    weather_info = get_weather(city, api_key)
    
    if "error" in weather_info:
        print(f"Error: {weather_info['error']}")
    else:
        print(f"Weather in {weather_info['city']}:")
        print(f"  Temperature: {weather_info['temperature']}°C")
        print(f"  Humidity: {weather_info['humidity']}%")
        print(f"  Pressure: {weather_info['pressure']} hPa")
        print(f"  Description: {weather_info['description']}")

if __name__ == "__main__":
    main()