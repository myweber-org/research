
import requests
import sys
from datetime import datetime, timedelta

def fetch_recent_issues(owner, repo, days=7, max_issues=10):
    """
    Fetch recent issues from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    params = {
        'state': 'all',
        'since': since_date,
        'per_page': max_issues,
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        issues = response.json()
        return issues
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        return []

def display_issues(issues):
    """
    Display issue details in a simple format.
    """
    if not issues:
        print("No issues found.")
        return

    for issue in issues:
        number = issue.get('number', 'N/A')
        title = issue.get('title', 'No Title')
        state = issue.get('state', 'unknown')
        user = issue.get('user', {}).get('login', 'Unknown')
        updated_at = issue.get('updated_at', '')
        if updated_at:
            updated_at = datetime.strptime(updated_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M')
        print(f"#{number} [{state.upper()}] {title}")
        print(f"   Author: {user}, Updated: {updated_at}")
        print()

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_github_issues.py <owner> <repo> [days] [max_issues]")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    max_issues = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    print(f"Fetching up to {max_issues} issues from {owner}/{repo} updated in the last {days} days...\n")
    issues = fetch_recent_issues(owner, repo, days, max_issues)
    display_issues(issues)

if __name__ == "__main__":
    main()