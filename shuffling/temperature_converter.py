
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def celsius_to_kelvin(celsius):
    return celsius + 273.15

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def fahrenheit_to_kelvin(fahrenheit):
    celsius = fahrenheit_to_celsius(fahrenheit)
    return celsius_to_kelvin(celsius)

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def kelvin_to_fahrenheit(kelvin):
    celsius = kelvin_to_celsius(kelvin)
    return celsius_to_fahrenheit(celsius)

def convert_temperature(value, from_unit, to_unit):
    units = ['C', 'F', 'K']
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Invalid unit. Use one of: {units}")
    
    if from_unit == to_unit:
        return value
    
    conversion_map = {
        ('C', 'F'): celsius_to_fahrenheit,
        ('C', 'K'): celsius_to_kelvin,
        ('F', 'C'): fahrenheit_to_celsius,
        ('F', 'K'): fahrenheit_to_kelvin,
        ('K', 'C'): kelvin_to_celsius,
        ('K', 'F'): kelvin_to_fahrenheit,
    }
    
    if (from_unit, to_unit) in conversion_map:
        return conversion_map[(from_unit, to_unit)](value)
    
    raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")

if __name__ == "__main__":
    print("Temperature Converter Examples:")
    print(f"25°C to Fahrenheit: {convert_temperature(25, 'C', 'F'):.2f}°F")
    print(f"100°F to Celsius: {convert_temperature(100, 'F', 'C'):.2f}°C")
    print(f"0°C to Kelvin: {convert_temperature(0, 'C', 'K'):.2f}K")
    print(f"300K to Fahrenheit: {convert_temperature(300, 'K', 'F'):.2f}°F")