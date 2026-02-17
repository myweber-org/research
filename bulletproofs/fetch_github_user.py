
import requests
import json

def fetch_github_user(username):
    """
    Fetch public details of a GitHub user.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        return {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "public_repos": user_data.get("public_repos"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "created_at": user_data.get("created_at")
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        details = fetch_github_user(username)
        if details:
            print(json.dumps(details, indent=2))
        else:
            print(f"Could not fetch details for user '{username}'.")