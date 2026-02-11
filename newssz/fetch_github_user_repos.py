
import requests
import sys

def fetch_user_repos(username, per_page=30, page=1):
    url = f"https://api.github.com/users/{username}/repos"
    params = {"per_page": per_page, "page": page}
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return

    for repo in repos:
        name = repo.get("name", "N/A")
        description = repo.get("description", "No description")
        stars = repo.get("stargazers_count", 0)
        print(f"{name}: {description} (Stars: {stars})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_user_repos.py <username> [per_page] [page]")
        sys.exit(1)

    username = sys.argv[1]
    per_page = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    page = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    repos = fetch_user_repos(username, per_page, page)
    if repos is not None:
        display_repos(repos)

if __name__ == "__main__":
    main()