
import requests
import time

def fetch_github_user(username, token=None):
    """
    Fetch GitHub user information from the GitHub API.
    
    Args:
        username (str): GitHub username to fetch.
        token (str, optional): GitHub personal access token for higher rate limits.
    
    Returns:
        dict: User data if successful, None otherwise.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check rate limit headers
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
        
        if remaining == 0:
            wait_time = reset_time - time.time()
            if wait_time > 0:
                print(f"Rate limit exceeded. Try again in {wait_time:.0f} seconds.")
                return None
        
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"User '{username}' not found.")
        elif e.response.status_code == 403:
            print("Rate limit exceeded or access forbidden.")
        else:
            print(f"HTTP error occurred: {e}")
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def display_user_info(user_data):
    """Display formatted user information."""
    if not user_data:
        return
    
    print(f"GitHub User: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    # Example usage
    user = fetch_github_user("octocat")
    display_user_info(user)