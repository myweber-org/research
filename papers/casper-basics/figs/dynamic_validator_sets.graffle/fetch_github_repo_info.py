
import requests
import json

def get_repo_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        info = {
            "name": data.get("name"),
            "full_name": data.get("full_name"),
            "description": data.get("description"),
            "html_url": data.get("html_url"),
            "stargazers_count": data.get("stargazers_count"),
            "forks_count": data.get("forks_count"),
            "open_issues_count": data.get("open_issues_count"),
            "language": data.get("language"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at")
        }
        return info
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository info: {e}")
        return None

def display_repo_info(info):
    if info:
        print(f"Repository: {info['full_name']}")
        print(f"Description: {info['description']}")
        print(f"URL: {info['html_url']}")
        print(f"Stars: {info['stargazers_count']}")
        print(f"Forks: {info['forks_count']}")
        print(f"Open Issues: {info['open_issues_count']}")
        print(f"Language: {info['language']}")
        print(f"Created: {info['created_at']}")
        print(f"Last Updated: {info['updated_at']}")
    else:
        print("No repository information available.")

if __name__ == "__main__":
    owner = "torvalds"
    repo = "linux"
    
    print(f"Fetching information for {owner}/{repo}...")
    repo_info = get_repo_info(owner, repo)
    display_repo_info(repo_info)