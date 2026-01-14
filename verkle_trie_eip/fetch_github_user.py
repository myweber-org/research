import requests
import sys

def get_github_user(username):
    """Fetch GitHub user information from the public API."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None

def display_user_info(user_data):
    """Display selected user information."""
    if not user_data:
        print("No user data to display.")
        return
    
    print(f"Username: {user_data.get('login', 'N/A')}")
    print(f"Name: {user_data.get('name', 'N/A')}")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Profile URL: {user_data.get('html_url', 'N/A')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_data = get_github_user(username)
    
    if user_data:
        display_user_info(user_data)
    else:
        print(f"Failed to fetch data for user '{username}'.")