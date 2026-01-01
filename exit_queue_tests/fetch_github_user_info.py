import requests
import sys

def get_github_user_info(username):
    """Fetch and display public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print(f"GitHub User: {data.get('login', 'N/A')}")
        print(f"Name: {data.get('name', 'N/A')}")
        print(f"Bio: {data.get('bio', 'N/A')}")
        print(f"Public Repos: {data.get('public_repos', 0)}")
        print(f"Followers: {data.get('followers', 0)}")
        print(f"Following: {data.get('following', 0)}")
        print(f"Profile URL: {data.get('html_url', 'N/A')}")
        
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user data: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    get_github_user_info(username)