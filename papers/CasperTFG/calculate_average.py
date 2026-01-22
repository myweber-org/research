
def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    return total / count

def main():
    sample_data = [10, 20, 30, 40, 50]
    result = calculate_average(sample_data)
    print(f"The average is: {result}")

if __name__ == "__main__":
    main()