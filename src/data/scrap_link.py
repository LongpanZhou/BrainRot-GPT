import praw
import time
from src.reddit import CLIENT_ID, CLIENT_SECRET, USER_AGENT

# List of brainrot terms
brainrot_terms = [
    "Skibidi", "Rizz", "Gyatt", "Goofy ahh", "Sigma grindset", "NPC behavior", "Based", "Cringe",
    "W", "L", "No cap", "Mid", "Bruh moment", "Ratio", "Bet", "Sus", "Drip", "Ayo", "Yeet", "Sheesh",
    "Bussin", "On god", "Malding", "Touch grass", "Glazing", "Cope", "Seethe", "Chad", "Doomer", "Zoomer",
    "Giga", "Vibe check", "BOZO", "Grimace shake", "Fumbled", "Certified", "Jit", "Delulu", "Simp", "Gobbler",
    "Pls", "Yap", "Pookie", "Moots", "Feral", "Shambles", "Lore", "Cannon event", "BFFR", "We ball"
]

# Initialize Reddit API client
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

# file name
FILE_NAME = "input.txt"

# Function to check if a post contains any brainrot terms
def contains_brainrot(text, terms):
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)

# Scrape Reddit for posts and return full /r/ links
def scrape_reddit_brainrot(terms, limit_per_term=100):
    post_links = set()
    subreddit = reddit.subreddit("all")

    for term in terms:
        print(f"Searching for posts with term: {term}")
        try:
            for submission in subreddit.search(term, limit=limit_per_term):
                if contains_brainrot(submission.title, terms) or \
                   (submission.selftext and contains_brainrot(submission.selftext, terms)):
                    # Use the full permalink format
                    full_url = f"https://www.reddit.com{submission.permalink}"
                    post_links.add(full_url)
            time.sleep(1)  # Pause between searches to avoid hitting API limits
        except Exception as e:
            print(f"Error searching for {term}: {e}")
            time.sleep(5)  # Wait longer if there's an error

    return list(post_links)

# Run the scraper
if __name__ == "__main__":
    # Scrape posts (limit of 100 per term to keep it manageable)
    links = scrape_reddit_brainrot(brainrot_terms, limit_per_term=100)

    # Print results
    print(f"Found {len(links)} unique posts with brainrot terms:")
    for i, link in enumerate(links, 1):
        print(f"{i}. {link}")

    # Optionally save to a file
    with open(FILE_NAME, "w", encoding="utf-8") as f:
        for link in links:
            f.write(f"{link}\n")
    print("Links saved to 'input.txt'")