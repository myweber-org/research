
import requests
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
        data = response.json()
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return
        
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        city_name = data['name']
        country = data['sys']['country']
        
        print(f"Weather in {city_name}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description.capitalize()}")
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except KeyError as e:
        print(f"Unexpected API response format: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    get_weather(api_key, city)