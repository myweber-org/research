
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    else:
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user(username)
    
    if user_info:
        print(f"Name: {user_info['name']}")
        print(f"Public Repos: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Account Created: {user_info['created_at']}")
    else:
        print("User not found or API error.")import requests
import time

def fetch_github_user(username):
    """
    Fetch public information for a given GitHub username.
    Returns a dictionary with user data or None if not found/error.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-Script"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check rate limit
        if response.status_code == 403 and 'rate limit' in response.text.lower():
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            wait_time = max(0, reset_time - int(time.time()))
            print(f"Rate limit exceeded. Try again in {wait_time} seconds.")
            return None
            
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"User '{username}' not found on GitHub.")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    return None

def display_user_info(user_data):
    """Display formatted user information."""
    if not user_data:
        return
    
    print(f"\nGitHub User: {user_data.get('login', 'N/A')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')[:100]}...")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Profile URL: {user_data.get('html_url', 'N/A')}")

if __name__ == "__main__":
    # Example usage
    username = input("Enter GitHub username: ").strip()
    if username:
        user_data = fetch_github_user(username)
        display_user_info(user_data)
    else:
        print("No username provided.")