import requests
import sys

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'name': data.get('name'),
            'public_repos': data.get('public_repos'),
            'followers': data.get('followers'),
            'following': data.get('following'),
            'created_at': data.get('created_at')
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user data: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_info = fetch_github_user(username)
    
    if user_info:
        print(f"GitHub User: {username}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Account Created: {user_info['created_at']}")

if __name__ == "__main__":
    main()