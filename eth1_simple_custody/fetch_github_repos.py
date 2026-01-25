import requests
import sys

def fetch_github_repos(username):
    """
    Fetch public repositories for a given GitHub username.
    Returns a list of repository names or an empty list on error.
    """
    url = f"https://api.github.com/users/{username}/repos"
    try:
        response = requests.get(url)
        response.raise_for_status()
        repos = response.json()
        return [repo['name'] for repo in repos]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return []
    except (KeyError, TypeError):
        print("Error parsing response data.", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repositories = fetch_github_repos(username)
    
    if repositories:
        print(f"Public repositories for user '{username}':")
        for repo in repositories:
            print(f"  - {repo}")
    else:
        print(f"No public repositories found for user '{username}' or an error occurred.")