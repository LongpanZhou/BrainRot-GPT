import re
import praw
from src.reddit import CLIENT_ID, CLIENT_SECRET, USER_AGENT

LINKS_FILE = "input.txt"
OUTPUT_FILE = "output.txt"

def binary_to_text(binary_sequence):
    """Convert binary string to text and print the process."""
    binary_values = binary_sequence.split(' ')
    text = ''

    for bv in binary_values:
        char = chr(int(bv, 2))  # Convert binary to character
        print(f"Binary: {bv} → Character: {char}")
        text += char

    return text


def main():
    reddit = praw.Reddit(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         user_agent=USER_AGENT)

    urls = set()
    with open(LINKS_FILE, 'r', encoding='utf-8', errors='ignore') as input_file:
        for line in input_file:
            urls.add(line.strip())

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        for i, url in enumerate(urls):
            submission_id = url.split('/')[-3]

            try:
                submission = reddit.submission(id=submission_id)
                submission.comments.replace_more(limit=0)

                for comment in submission.comments.list():
                    if comment.body in ['[deleted]', '[removed]']:
                        continue

                    comment_text = re.sub(r'!\[.*?\]\(.*?\)|\[.*?\]\(.*?\)', '', comment.body)
                    comment_text = re.sub(r'http[s]?://\S+', '', comment_text)

                    comment_text = comment_text.strip()
                    comment_text = re.sub(r'\n\s*\n', '\n', comment_text)

                    if comment_text:
                        print(f"Comment: {comment_text} | Length: {len(comment_text)}")

                        try:
                            binary_comment = re.findall(r'[01]{8}', comment_text)
                            if binary_comment:
                                decoded_comment = binary_to_text(" ".join(binary_comment))
                                print(f"Decoded Binary: {decoded_comment}")
                        except Exception as e:
                            print(f"Error processing binary in comment: {e}")

                        output_file.write(f"[BOS] {comment_text.lower()} [EOS]\n")

            except praw.exceptions.PRAWException as e:
                print(f"Error processing {url}: {e}")
                continue

            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue

            print(f"Processed {i + 1}/{len(urls)}")


if __name__ == "__main__":
    main()
