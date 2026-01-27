import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_data = get_github_user(username)
    
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
    else:
        print(f"Failed to fetch data for user: {username}")