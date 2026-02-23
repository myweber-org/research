import requests
import sys

def get_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'company': user_data.get('company'),
            'blog': user_data.get('blog'),
            'location': user_data.get('location'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following')
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching data: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def display_user_info(info):
    """
    Display the fetched user information in a readable format.
    """
    if not info:
        print("No information to display.")
        return
    print("GitHub User Information:")
    print("-" * 30)
    for key, value in info.items():
        if value is not None:
            print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            print(f"{key.replace('_', ' ').title()}: Not provided")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    user_info = get_github_user_info(username)
    display_user_info(user_info)