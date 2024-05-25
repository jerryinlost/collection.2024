import os
import argparse
import re
import arxiv


def setup_argument_parser():
    """
    Sets up and returns an argument parser for the script.

    Returns:
    An argparse.ArgumentParser instance configured for the script.
    """
    parser = argparse.ArgumentParser(
        description="Download articles from arXiv.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "url_or_id",
        type=str,
        help="The URL or ID of the arXiv article.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="./",
        help="The directory where the article will be downloaded.",
    )
    parser.add_argument(
        "-s",
        "--source",
        action="store_true",
        help="Also download the source files of the article.",
    )

    args = parser.parse_args()

    if args.url_or_id is None:
        print("Error: Please provide a URL or ID for the article.\n")
        parser.print_help()
        exit(1)

    return args


def is_url(input_string):
    """
    Determines if the input string is a URL.

    Returns:
    True if the string is a URL, False otherwise.
    """
    return bool(re.match(r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/\d+\.\d+(v\d+)?(\.pdf)?$", input_string))


def parse_id(input_string):
    """
    Extracts the arXiv ID from a given input string, which can be either an arXiv ID or a URL.

    Parameters:
    - input_string: The input string that is either an arXiv ID or a URL.

    Returns:
    The extracted arXiv ID if the input is valid, otherwise None.
    """
    # Pattern to match a direct arXiv ID
    id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
    if id_pattern.match(input_string):
        return input_string

    # Pattern to match an arXiv URL and extract the ID
    url_pattern = re.compile(r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$")
    url_match = url_pattern.match(input_string)
    if url_match:
        return url_match.group(2) + (url_match.group(3) if url_match.group(3) else "")

    # If the input does not match any of the expected formats
    print("Error: The provided input does not match the expected URL or ID format.")
    exit(1)

def main():
    args = setup_argument_parser()

    # Determine if input is a URL or ID
    article_id = parse_id(args.url_or_id)

    # Proceed with the download
    directory = args.directory
    os.makedirs(directory, exist_ok=True)
    search_result = arxiv.Client().results(arxiv.Search(id_list=[article_id]))

    if article := next(search_result):
        print(f'Starting download of article: "{article.title}" ({article_id})')
        pdf_path = article.download_pdf(dirpath=directory)
        print(f"Download finished! Result saved at:\n{pdf_path}")

        if args.source:
            print(f'Starting download of article source files: "{article.title}" ({article_id})')
            article.download_source(dirpath=directory)
    else:
        print("Article not found.")


if __name__ == "__main__":
    main()
