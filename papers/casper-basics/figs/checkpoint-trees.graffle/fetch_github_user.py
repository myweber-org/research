import requests

def fetch_github_user(username):
    """Fetch public details of a GitHub user."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"User '{username}' not found or API error."}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        username = sys.argv[1]
        user_data = fetch_github_user(username)
        print(user_data)
    else:
        print("Please provide a GitHub username as an argument.")