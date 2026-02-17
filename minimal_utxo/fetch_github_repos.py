import requests
import sys

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Unable to fetch repositories for {username}")
        return []
    
    repos = response.json()
    sorted_repos = sorted(repos, key=lambda x: x['stargazers_count'], reverse=True)
    return sorted_repos

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    
    print(f"{'Repository Name':<40} {'Stars':<10} {'Language':<15}")
    print("-" * 70)
    for repo in repos:
        name = repo['name'][:38] + '..' if len(repo['name']) > 40 else repo['name']
        stars = repo['stargazers_count']
        language = repo['language'] or 'N/A'
        print(f"{name:<40} {stars:<10} {language:<15}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = get_github_repos(username)
    display_repos(repos)