import requests
import json

def fetch_github_repos(username):
    """
    Fetches the public repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_list = []
        for repo in repos:
            repo_info = {
                'name': repo['name'],
                'description': repo['description'],
                'url': repo['html_url'],
                'stars': repo['stargazers_count']
            }
            repo_list.append(repo_info)
        return repo_list
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return None

def display_repos(repos):
    """
    Displays the list of repositories in a formatted way.
    """
    if not repos:
        print("No repositories to display.")
        return
    
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"Description: {repo['description']}")
        print(f"URL: {repo['url']}")
        print(f"Stars: {repo['stars']}")
        print("-" * 40)

if __name__ == "__main__":
    username = input("Enter a GitHub username: ")
    repositories = fetch_github_repos(username)
    if repositories:
        display_repos(repositories)
import requests
import sys

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
    else:
        print(f"Error: Unable to fetch repositories for user '{username}'")
        print(f"Status Code: {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    get_github_repos(username)