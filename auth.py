import yaml

def create_url(user):
    """Creates request URL for the specified username"""
    username = user
    url = f"https://api.twitter.com/2/tweets/search/recent?query=from:{username}&max_results=100"
    return url

def get_token():
    with open("config.yaml") as f:
        data = yaml.safe_load(f)
        return data['twitter_api']['bearer_token']