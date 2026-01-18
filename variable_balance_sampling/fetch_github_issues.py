
import requests
import sys

def fetch_issues(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "open"}
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        issues = response.json()
        
        if not issues:
            print(f"No open issues found for {owner}/{repo}.")
            return
        
        print(f"Open issues for {owner}/{repo}:")
        for issue in issues:
            print(f"  #{issue['number']}: {issue['title']}")
            print(f"     URL: {issue['html_url']}")
            print()
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_issues.py <owner> <repo>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    fetch_issues(owner, repo)