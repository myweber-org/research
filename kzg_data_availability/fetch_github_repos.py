import requests

def fetch_user_repos(username, per_page=10, page=1):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'per_page': per_page,
        'page': page
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"Description: {repo['description'] or 'No description'}")
        print(f"URL: {repo['html_url']}")
        print(f"Stars: {repo['stargazers_count']}")
        print(f"Forks: {repo['forks_count']}")
        print("-" * 40)

def main():
    username = input("Enter GitHub username: ")
    per_page = int(input("Repos per page (default 10): ") or 10)
    page = int(input("Page number (default 1): ") or 1)
    
    try:
        repos = fetch_user_repos(username, per_page, page)
        display_repos(repos)
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching repositories: {e}")
    except ValueError:
        print("Invalid input for per_page or page.")

if __name__ == "__main__":
    main()