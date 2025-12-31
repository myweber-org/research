import requests

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Username: {user_data['login']}")
        print(f"Name: {user_data.get('name', 'N/A')}")
        print(f"Public Repos: {user_data['public_repos']}")
        print(f"Followers: {user_data['followers']}")
        print(f"Following: {user_data['following']}")
        print(f"Profile URL: {user_data['html_url']}")
    else:
        print(f"Error: Unable to fetch user info (Status: {response.status_code})")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    get_github_user_info(username)