import requests

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {'error': f'User "{username}" not found.'}
        else:
            return {'error': f'HTTP error occurred: {e}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Request failed: {e}'}

def display_user_info(info):
    """Display the fetched user information."""
    if 'error' in info:
        print(f"Error: {info['error']}")
    else:
        print(f"GitHub User: {info['login']}")
        print(f"Name: {info['name']}")
        print(f"Public Repositories: {info['public_repos']}")
        print(f"Followers: {info['followers']}")
        print(f"Following: {info['following']}")
        print(f"Profile URL: {info['html_url']}")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        user_info = fetch_github_user(username)
        display_user_info(user_info)
    else:
        print("No username provided.")