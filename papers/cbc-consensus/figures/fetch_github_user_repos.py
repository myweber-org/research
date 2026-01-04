import requests
import argparse
from datetime import datetime

def fetch_user_repositories(username, sort_by='updated', order='desc'):
    """
    Fetch repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'sort': sort_by,
        'direction': order,
        'per_page': 100
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No repositories found for user: {username}")
            return []
            
        return repos
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []

def display_repositories(repos, max_display=10):
    """
    Display repository information in a formatted way.
    """
    if not repos:
        return
    
    print(f"\nFound {len(repos)} repositories:")
    print("-" * 80)
    
    for i, repo in enumerate(repos[:max_display]):
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        updated = repo.get('updated_at', 'N/A')
        
        if updated != 'N/A':
            updated_date = datetime.fromisoformat(updated.replace('Z', '+00:00'))
            updated_str = updated_date.strftime('%Y-%m-%d')
        else:
            updated_str = 'N/A'
        
        print(f"{i+1}. {name}")
        print(f"   Description: {description}")
        print(f"   Stars: {stars} | Forks: {forks} | Updated: {updated_str}")
        print(f"   URL: {repo.get('html_url', 'N/A')}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'],
                       default='updated', help='Sort repositories by field')
    parser.add_argument('--order', choices=['asc', 'desc'],
                       default='desc', help='Sort order')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of repositories to display')
    
    args = parser.parse_args()
    
    repos = fetch_user_repositories(args.username, args.sort, args.order)
    
    if repos:
        display_repositories(repos, args.limit)
        
        total_stars = sum(repo.get('stargazers_count', 0) for repo in repos)
        total_forks = sum(repo.get('forks_count', 0) for repo in repos)
        
        print(f"\nSummary for {args.username}:")
        print(f"Total repositories: {len(repos)}")
        print(f"Total stars: {total_stars}")
        print(f"Total forks: {total_forks}")

if __name__ == "__main__":
    main()