import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Union

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
import requests

from phase_1_site_detail_fetcher import run_pipeline_phase_1
from phase_2_category_reducer import run_pipeline_phase_2
from phase_3_subcategory_classifier import run_pipeline_phase_3

load_dotenv(find_dotenv())
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.__getattribute__(str(os.getenv("LOG_LEVEL", "INFO"))))


@dataclass
class Bookmark:
    title: str
    url: str


@dataclass
class Category:
    category: str
    subcategory: str


def parse_chrome_bookmarks(html_file_path) -> List[Bookmark]:
    """
    Parse a Chrome bookmarks exported HTML file.

    Args:
        html_file_path (str): The path to the Chrome bookmarks HTML file.

    Returns:
        list: A list of dictionaries, each containing 'title' and 'url' of a bookmark.
    """
    # Open and read the HTML file
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <a> tags, as these represent the bookmarks
    bookmark_tags = soup.find_all("a")

    # Extract the title and URL from each <a> tag
    bookmarks = []
    for tag in bookmark_tags:
        title = tag.text
        url = tag.get("href")
        bookmarks.append(Bookmark(title, url))
    return bookmarks


def create_bookmark_html(bookmarks_dict: dict):
    html_start = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
<DT><H3 ADD_DATE="1725003099" LAST_MODIFIED="1725027983" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks bar</H3>
<DL><p>\n"""

    def process_folder(
        folder_name: str,
        folder_content: Union[List[Bookmark], Dict[str, List[Bookmark]]],
    ) -> str:
        html = ""
        date = int(time.time())
        if folder_name and folder_name != "":
            html += f'<DT><H3 ADD_DATE="{date}" LAST_MODIFIED="{date}">{folder_name}</H3>\n<DL><p>\n'

        if isinstance(folder_content, dict):
            for sub_folder_name, bookmarks in folder_content.items():
                if sub_folder_name == "_CHILDREN_":
                    html += (
                        "\n".join(
                            [
                                f'<DT><A HREF="{bookmark.url}" ADD_DATE="{date}" LAST_VISIT="{date}" LAST_MODIFIED="{date}">{bookmark.title}</A>'
                                for bookmark in bookmarks
                            ]
                        )
                        + "\n</DL><p>\n"
                    )
                else:
                    html += process_folder(sub_folder_name, bookmarks)
        elif isinstance(folder_content, list):
            html += "\n".join([f'<DT><A HREF="{bookmark.url}">{bookmark.title}</A>' for bookmark in folder_content]) + "\n</DL><p>\n"

        return html

    html_content = ""
    for folder_name, folder_content in bookmarks_dict.items():
        html_content += process_folder(folder_name, folder_content)

    html_output = html_start + html_content + "\n</DL><p>\n</DL><p>"

    with open("bookmarks.html", "w", encoding="utf-8") as file:
        file.write(html_output)


if __name__ == "__main__":
    logger.info("Starting")
    parser = argparse.ArgumentParser(description="Categorize Browser Bookmarks using AI.")
    parser.add_argument("-f", "--file", help="Path to the exported bookmarks HTML file.", required=True)
    parser.add_argument("--ollama", help="Address to your Ollama service.", default=None)
    parser.add_argument("--model", help="Ollama model to use for inference. See: https://ollama.com/library", default=None)
    parser.add_argument("--chrome", help="Address to your Chrome Debugging Port.", default=None)

    args = parser.parse_args()

    html_file_path = args.file
    ollama_host = "http://" + args.ollama if args.ollama is not None else "http://localhost:11434"
    ollama_model = args.model if args.model is not None else "llama3.1:latest"
    chrome_host = "http://" + args.chrome if args.chrome is not None else "http://localhost:9222"

    logger.info(f"Testing Chrome connectivity @ {chrome_host}")
    try:
        resp = requests.get(f"{chrome_host}/json/version")
        if resp.status_code != 200:
            raise Exception("Failed to connect to Chrome Debugging Port.")
        logger.info("Chrome connected successfully!")
    except Exception:
        logger.exception("Failed to connect to Chrome Debugging Port")
        exit()

    logger.info(f"Testing Ollama model connectivity and inference @ {ollama_host}")
    try:
        resp = requests.get(f"{ollama_host}/v1/models")
        if resp.status_code != 200:
            raise Exception("Failed to connect to Ollama service.")
        logger.info("Connected successfully!")
        resp = requests.post(f"{ollama_host}/api/generate", json={"model": ollama_model, "prompt": "Hello"})
        if resp.status_code != 200:
            raise Exception("Failed to run inference with Ollama service.")
        logger.info("Inference successful!")
        logger.info(f"Ollama service is up and running @ {ollama_host}")
    except Exception:
        logger.exception("Failed to connect to Ollama service.")
        exit()

    logger.info(f"Reading bookmarks file @ {html_file_path}")
    input_bookmarks = parse_chrome_bookmarks(html_file_path)
    logger.info(f"Found {len(input_bookmarks)} bookmarks")

    phase_1 = {}
    bookmarks_by_category: Dict[str, List[Bookmark]] = {}
    # check if phase_1_analysis.json exists, then load it to phase_1 and construct bookmarks by category
    if os.path.exists("phase_1_analysis.json"):
        with open("phase_1_analysis.json", "r") as file:
            phase_1 = json.load(file)
            for url, bookmark_data in phase_1.items():
                if "predicted_category" in bookmark_data:
                    bmrk = Bookmark(
                        title=bookmark_data["predicted_title"],
                        url=bookmark_data["url"],
                    )
                    bookmarks_by_category.setdefault(bookmark_data["predicted_category"], []).append(bmrk)
        # fetch websites
        logger.info(f"Re-processing bookmarks_by_category['Dead'] - Total: {len(bookmarks_by_category['Dead'])}")
        dead_bookmarks_found = []
        for idx, bookmark in enumerate(bookmarks_by_category["Dead"]):
            logger.info(f"Checking bookmark {bookmark.url} {idx}/{len(bookmarks_by_category['Dead'])}")
            try:
                response = asyncio.run(run_pipeline_phase_1(bookmark.url, bookmark.title, chrome_host, ollama_host, ollama_model))
                time.sleep(1)  # Add a delay to avoid overwhelming the webservers :-)
                response["url"] = bookmark.url
                if isinstance(response["predicted_category"], list):
                    response["predicted_category"] = str(response["predicted_category"][0])

            except Exception:
                logger.exception(f"Could not process pipeline for {bookmark.title} {bookmark.url}")
                response = {
                    "predicted_title": bookmark.title,
                    "url": bookmark.url,
                    "predicted_category": "Dead",
                }
            if response["predicted_category"] != "Dead":
                phase_1[response["url"]] = response
                bookmarks_by_category.setdefault(response["predicted_category"], []).append(bookmark)
                dead_bookmarks_found.append(bookmark)
        for bookmark in dead_bookmarks_found:
            bookmarks_by_category["Dead"].remove(bookmark)
        # save bookmarks_by_category to json file
        with open("phase_1_analysis.json", "w") as f:
            json.dump(phase_1, f, indent=4)
            logger.info("Saved phase 1 analysis to json file")

    else:
        # fetch websites
        for idx, bookmark in enumerate(input_bookmarks):
            if bookmark.url in phase_1:
                continue
            else:
                logger.debug(f"Already processed URL - Skipping {bookmark.url}")
            logger.info(f"Checking bookmark {bookmark.url} {idx}/{len(input_bookmarks)}")
            # if idx > 125: # If you want to test with only first few bookmarks
            #     break
            try:
                response = asyncio.run(run_pipeline_phase_1(bookmark.url, bookmark.title, chrome_host, ollama_host, ollama_model))
                time.sleep(2)  # Add a delay to avoid overwhelming the chrome browser :-)
                response["url"] = bookmark.url
                if isinstance(response["predicted_category"], list):
                    response["predicted_category"] = str(response["predicted_category"][0])

            except Exception:
                logger.exception(f"Could not process pipeline for {bookmark.title} {bookmark.url}")
                response = {
                    "predicted_title": bookmark.title,
                    "url": bookmark.url,
                    "predicted_category": "Dead",
                }
            phase_1[response["url"]] = response
            bookmarks_by_category.setdefault(str(response["predicted_category"]), []).append(bookmark)
        # save bookmarks_by_category to json file
        with open("phase_1_analysis.json", "w") as f:
            json.dump(phase_1, f, indent=4)
            logger.info("Saved phase 1 analysis to json file")

    # pprint(bookmarks_by_category)
    original_count = 0
    for category in bookmarks_by_category:
        original_count += len(bookmarks_by_category[category])
        logger.info(f"Category {category}: {len(bookmarks_by_category[category])}")

    logger.info("Cleaning up categories")
    new_categories = asyncio.run(run_pipeline_phase_2(bookmarks_by_category, ollama_host, ollama_model))

    logger.info("New categories:")
    # pprint(new_categories)
    reduced_category_count = 0
    for category in new_categories:
        reduced_category_count += len(new_categories[category])
        logger.info(f"Category {category}: {len(new_categories[category])}")

    logger.info(f"Bookmark counts => Phase 1: {original_count}, Phase 2: {reduced_category_count}")

    logger.info("Sub-dividing crowded categories")
    for category in new_categories:
        if category == "Dead":
            continue
        if len(new_categories[category]) > 5:
            logger.info(f"Category {category} has more than 10 bookmarks, subdividing")
            bookmarks_by_sub_category: Dict[str, List[Bookmark]] = {}
            for bookmark in new_categories[category]:
                enhanced_prediction = asyncio.run(run_pipeline_phase_3(phase_1[bookmark.url], ollama_host, ollama_model))
                bookmarks_by_sub_category.setdefault(str(enhanced_prediction["predicted_sub_category"]), []).append(bookmark)
            # logger.info(f"New subcategories for {category}")
            # pprint(bookmarks_by_sub_category.keys())
            logger.info("\tReducing subcategories")
            new_sub_categories = asyncio.run(run_pipeline_phase_2(bookmarks_by_sub_category, ollama_host, ollama_model))
            min_sub_category_bookmark_count = 3
            # if a subcategory has 1 or 2 bookmarks only, then just leave it in the main category
            for sub_category in [key for key in new_sub_categories.keys()]:
                if len(new_sub_categories[sub_category]) >= min_sub_category_bookmark_count:
                    continue
                else:
                    logger.info(
                        f"Sub-category {sub_category} has only {len(new_sub_categories[sub_category])} bookmarks, moving to main category _CHILDREN_"
                    )
                    # move the bookmark to the main category
                    for bookmark in new_sub_categories[sub_category]:
                        new_sub_categories.setdefault("_CHILDREN_", []).append(bookmark)
                    del new_sub_categories[sub_category]

            # logger.info(f"New reduced subcategories for {category}:")
            # pprint(new_sub_categories)
            reduced_category_count = 0
            for sub_category in new_sub_categories:
                reduced_category_count += len(new_sub_categories[sub_category])
                logger.info(f"Category {category}: {len(new_sub_categories[sub_category])}")
            logger.info(f"Bookmark {category} counts => Phase 2: {len(new_categories[category])}, Phase 3: {reduced_category_count}")
            new_categories[category] = new_sub_categories
    logger.info("Final bookmarks")
    for category in new_categories:
        # if new_categories[category] type of list, convert to dict with key "_CHILDREN_"
        if isinstance(new_categories[category], list):
            new_categories[category] = {"_CHILDREN_": new_categories[category]}
    create_bookmark_html(new_categories)
    logger.info("Finished - Produced enhanced bookmarks file, ready for import")
