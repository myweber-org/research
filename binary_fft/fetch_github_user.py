import requests

def fetch_github_user(username):
    """Fetch user information from GitHub API."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print(f"Error: Unable to fetch user data. Status code: {response.status_code}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    fetch_github_user(username)