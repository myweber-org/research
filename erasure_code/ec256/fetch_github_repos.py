import requests

def fetch_github_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return

    print(f"Found {len(repos)} repositories:")
    for idx, repo in enumerate(repos, start=1):
        print(f"{idx}. {repo['name']}")
        print(f"   Description: {repo['description'] or 'No description'}")
        print(f"   Language: {repo['language'] or 'Not specified'}")
        print(f"   Stars: {repo['stargazers_count']}")
        print(f"   URL: {repo['html_url']}")
        print()

def main():
    username = input("Enter GitHub username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return

    page = 1
    while True:
        repos = fetch_github_repositories(username, page=page)
        display_repositories(repos)

        if len(repos) < 30:
            print("No more repositories.")
            break

        cont = input("Fetch next page? (y/n): ").strip().lower()
        if cont != 'y':
            break
        page += 1

if __name__ == "__main__":
    main()