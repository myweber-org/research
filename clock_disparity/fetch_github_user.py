import requests
import time

def fetch_github_user(username, token=None):
    """
    Fetch public information for a GitHub user.

    Args:
        username (str): GitHub username.
        token (str, optional): GitHub personal access token for higher rate limits.

    Returns:
        dict: User data if successful, None otherwise.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Check remaining rate limit
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))

        if remaining == 0:
            wait_time = reset_time - time.time()
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds.")
                time.sleep(wait_time + 1)
                return fetch_github_user(username, token)

        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"User '{username}' not found.")
        elif e.response.status_code == 403:
            print("Rate limit exceeded. Consider using an authentication token.")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return None

if __name__ == "__main__":
    # Example usage
    user_data = fetch_github_user("octocat")
    if user_data:
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")