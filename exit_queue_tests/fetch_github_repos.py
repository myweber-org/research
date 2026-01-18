import requests
import argparse
import sys

def fetch_repositories(username, sort_by='updated', order='desc'):
    """
    Fetch repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {'sort': sort_by, 'direction': order}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repositories(repos, max_repos=10):
    """
    Display repository information.
    """
    if not repos:
        print("No repositories found.")
        return
    
    print(f"Found {len(repos)} repositories. Displaying up to {max_repos}:\n")
    for idx, repo in enumerate(repos[:max_repos], 1):
        print(f"{idx}. {repo['name']}")
        print(f"   Description: {repo.get('description', 'No description')}")
        print(f"   Stars: {repo['stargazers_count']}")
        print(f"   Updated: {repo['updated_at']}")
        print(f"   URL: {repo['html_url']}\n")

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub repositories for a user.')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'],
                        default='updated', help='Sort repositories by field')
    parser.add_argument('--order', choices=['asc', 'desc'], default='desc',
                        help='Order of sorting (ascending or descending)')
    parser.add_argument('--max', type=int, default=10,
                        help='Maximum number of repositories to display')
    
    args = parser.parse_args()
    
    repos = fetch_repositories(args.username, args.sort, args.order)
    if repos is not None:
        display_repositories(repos, args.max)

if __name__ == "__main__":
    main()