import requests

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'location': user_data.get('location'),
            'blog': user_data.get('blog')
        }
    else:
        return None

def display_user_info(username, info):
    if info:
        print(f"GitHub User: {username}")
        print(f"Name: {info['name']}")
        print(f"Public Repositories: {info['public_repos']}")
        print(f"Followers: {info['followers']}")
        print(f"Following: {info['following']}")
        print(f"Location: {info['location']}")
        print(f"Blog: {info['blog']}")
    else:
        print(f"User '{username}' not found or API error.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user_info(username)
    display_user_info(username, user_info)