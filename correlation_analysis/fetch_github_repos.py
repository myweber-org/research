
import requests
import argparse
import sys

def get_user_repositories(username, sort_by='updated', order='desc'):
    """
    Fetch repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'sort': sort_by,
        'direction': order,
        'per_page': 100
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

def display_repositories(repos, max_count=10):
    """
    Display repository information.
    """
    if not repos:
        print("No repositories found or an error occurred.")
        return

    print(f"\nFound {len(repos)} repositories. Showing up to {max_count}:\n")
    print(f"{'Name':<30} {'Stars':<8} {'Updated':<20}")
    print("-" * 60)

    for idx, repo in enumerate(repos[:max_count]):
        name = repo.get('name', 'N/A')[:28]
        stars = repo.get('stargazers_count', 0)
        updated = repo.get('updated_at', 'N/A')[:10]
        print(f"{name:<30} {stars:<8} {updated:<20}")

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories.')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'],
                        default='updated', help='Sort repositories by field')
    parser.add_argument('--order', choices=['asc', 'desc'], default='desc',
                        help='Sort order')
    parser.add_argument('--max', type=int, default=10,
                        help='Maximum number of repositories to display')

    args = parser.parse_args()

    repos = get_user_repositories(args.username, args.sort, args.order)
    display_repositories(repos, args.max)

if __name__ == "__main__":
    main()