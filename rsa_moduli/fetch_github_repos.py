import requests
import sys

def fetch_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Repo: {repo['name']}")
            print(f"  Description: {repo['description']}")
            print(f"  URL: {repo['html_url']}")
            print(f"  Stars: {repo['stargazers_count']}")
            print()
    else:
        print(f"Failed to fetch repositories for user '{username}'. Status code: {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    fetch_github_repos(sys.argv[1])