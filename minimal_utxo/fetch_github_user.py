
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'name': data.get('name'),
            'login': data.get('login'),
            'public_repos': data.get('public_repos'),
            'followers': data.get('followers'),
            'following': data.get('following'),
            'created_at': data.get('created_at')
        }
    else:
        return None

def display_user_info(user_data):
    if user_data:
        print(f"Username: {user_data['login']}")
        print(f"Name: {user_data['name']}")
        print(f"Public Repositories: {user_data['public_repos']}")
        print(f"Followers: {user_data['followers']}")
        print(f"Following: {user_data['following']}")
        print(f"Account Created: {user_data['created_at']}")
    else:
        print("User not found or API request failed.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user(username)
    display_user_info(user_info)