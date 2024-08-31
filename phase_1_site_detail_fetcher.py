import json
import logging
import os
import re
import urllib.parse as up
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from lxml import html
from newspaper import fulltext

# from WebsiteFetcher import WbsiteFetcher
from playwright.async_api import async_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

load_dotenv(find_dotenv())
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.__getattribute__(str(os.getenv("LOG_LEVEL", "INFO"))))


def extract_description(html_content):
    """
    given the <head> html of a webpage, this function extracts the meta description and keywords.
    """
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all meta tags with name 'description' and 'keywords'
    # Extract the content of these meta tags if they exist
    try:
        description_meta = soup.find("meta", attrs={"name": "description"})
        description = description_meta["content"] if description_meta else None
    except Exception:
        logger.exception("Error extracting meta description")
        description = None

    try:
        keywords_meta = soup.find("meta", attrs={"name": "keywords"})
        keywords = keywords_meta["content"] if keywords_meta else None
    except Exception:
        logger.exception("Error extracting meta keywords")
        keywords = None

    return {"description": description, "keywords": keywords}


def extract_navbar_text(html_content):
    # Parse the HTML content
    tree = html.fromstring(html_content)

    # Attempt to find the <nav> element
    navbar = tree.xpath("//nav")

    if not navbar:
        # If <nav> isn't found, try looking for a <ul> with a class or id that suggests it's a navbar
        navbar = tree.xpath('//ul[contains(@class, "navbar") or contains(@id, "navbar")]')

    if not navbar:
        # If still not found, return an appropriate message
        return ""

    # Extracting text from the navbar
    navbar_items = [x.strip() for x in navbar[0].xpath(".//text()") if x.strip() != "" and "{" not in x]
    navbar_text = ",".join(navbar_items).strip()

    # Remove extra spaces
    navbar_text = re.sub(r"\s+", " ", navbar_text)

    return navbar_text


class MyOllamaGenerator(OllamaGenerator):
    def _create_json_payload(self, prompt: str, stream: bool, generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns a dictionary of JSON arguments for a POST request to an Ollama service.
        """
        generation_kwargs = generation_kwargs or {}
        return {
            "prompt": prompt,
            "model": self.model,
            "stream": stream,
            "raw": self.raw,
            "template": self.template,
            "system": self.system_prompt,
            "options": generation_kwargs,
            "format": "json",
        }


async def run_pipeline_phase_1(url, user_title, chrome_host, ollama_host, ollama_model):
    site_text = None
    # Connect to chrome debug port
    # headless-ubuntu
    response = requests.get(f"{chrome_host}/json/version")
    logger.info(f"response {response}")
    web_socket_debugger_url = response.json()["webSocketDebuggerUrl"]
    logger.info(f"Starting playwright connection to: {web_socket_debugger_url}")
    page_title = url
    parsed = up.urlparse(url)
    domain = parsed.netloc
    # Use Playwright to connect to the existing Chrome instance
    navbar_text = None
    website_html = None
    dead_bookmark_query = """
        Analyse the given website text carefully.
        Is the website operating normally/alive? Or is it dead? Dead website is when any of the below is present:
            - 404 error, 
            - Page Not Found, 
            - Domain Parking page, 
            - Domain is for sale, 
            - Domain still to be configured, 
            - DNS records need to be configured, 
            - Message for domain owner,
            - Account not paid.

        Respond in JSON:
        {
            status: alive | dead
        }
        """

    prepipe_llm = MyOllamaGenerator(
        model=ollama_model,
        url=f"{ollama_host}/api/generate",
        generation_kwargs={
            "num_predict": 150,
            "temperature": 0.75,
        },
    )
    prepipe = Pipeline()
    with open("prompts/system_prompt.jinja") as template_file:
        system_prompt = template_file.read()  # Read the whole file as string into variable

    with open("prompts/dead_alive_template.jinja") as template_file:
        dead_alive_template = template_file.read()  # Read the whole file as string into variable
    # with open("prompts/category_checker_template.jinja") as template_file:
    #     category_checker_template = template_file.read()  # Read the whole file as string into variable

    prepipe.add_component("prompt_deadalivebuilder", PromptBuilder(template=dead_alive_template))
    prepipe.add_component("prepipe_llm", prepipe_llm)
    prepipe.connect("prompt_deadalivebuilder", "prepipe_llm")
    description_keywords = {}
    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp(web_socket_debugger_url)
            context = browser.contexts[0]
            page = context.pages[0]
            await page.goto(url, timeout=7000)
            page_title = await page.title()
            logger.debug(f"Tab Title: {page_title}")
            description_keywords = extract_description(await page.inner_html("head"))
            logger.debug(f"HEAD Description Keywords: {description_keywords}")
            navbar_text = extract_navbar_text(await page.inner_html("body"))
            logger.debug(f"Navbar Text: {navbar_text}")
            try:
                website_html = await page.inner_html("html")
                site_text = fulltext(website_html)
            except Exception:
                # logger.exception('could not do fulltext')
                site_text = website_html

            result = prepipe.run(
                {
                    "prompt_deadalivebuilder": {
                        "query": dead_bookmark_query,
                        "website_text": site_text if site_text else "",
                    }
                }
            )
            result = json.loads(result["prepipe_llm"]["replies"][0])

            if result["status"] == "dead":
                logger.info(f"{url} - Dead bookmark!")
                return {
                    "page_title": user_title,
                    "predicted_title": user_title,
                    "predicted_category": "Dead",
                }
        except PlaywrightTimeoutError:
            logger.info(f"Page {url} timedout, using blank page title and navbar text")
            page_title = ""
            navbar_text = ""
        except Exception:
            logger.exception(f"URL {url} could not load - not processing - classifying as 'Dead'")
            return {
                "page_title": user_title,
                "predicted_title": user_title,
                "predicted_category": "Dead",
            }

    query = """You are given a URL, (%s), the Page Title of the website is (%s). The domain is (%s).
                %s
                %s
                %s
               Based on this information, answer the following questions:
                1. What is a good title for this URL?
                    - Be short and concise (2 words maximum)
                    - Do not use vague words like "Home", "Page" or "Startseite".
                    - Always start the title with a ref to the domain, followed by a couple of relevant words regarding the service provided.
                2. In what main category would you categorize this website's service?
                    - Choose a simple generic high-level content category from the following list:
                        -   "News", "Social Media", "E-commerce", "Marketplace", "Education", "Finance", "Health", "Travel", "Entertainment", "Technology", "Science", "Art", "Sports", "Government", "Nonprofit", "Real Estate", "Automotive", "Jobs", "Pets", "Agriculture", "Music", "Fashion", "Food & Drink", "Business", "Weather", "Architecture", "Design", "Marketing", "Sales", "Legal", "Insurance", "Environment", "Community", "Tools", "Games", "Communication", "Other"
                3. Give a short concise summary description of this website's service.
                    - Keep it short and concise (2-3 sentences max)

                For each field:
                    - Do not use irrelevant numbers.
                    - Use proper Capitalization
                    - Avoid special characters or numbers unless necessary for clarity.

                Output your answers as a JSON object with the following structure:
                {
                    "title": str,
                    "category": str,
                    "description": str
                }
                """ % (
        url,
        page_title,
        domain,
        (
            f"[important] The given website description is: ({description_keywords['description']})."
            if description_keywords.get("description")
            else ""
        ),
        f"The given website keywords are: ({description_keywords['keywords']})." if description_keywords.get("keywords") else "",
        f"The website navbar text is: ({navbar_text})." if navbar_text else "",
    )

    logger.info(f"Connecting to {ollama_host} using model {ollama_model}")
    predictor_generator = MyOllamaGenerator(
        model=ollama_model,
        url=f"{ollama_host}/api/generate",
        generation_kwargs={
            "num_predict": 150,
            "temperature": 0.75,
        },
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=system_prompt))
    pipe.add_component("predictor_llm", predictor_generator)

    pipe.connect("prompt_builder", "predictor_llm")

    logger.debug(f"Starting pipeline for {url}")
    result = pipe.run(
        {
            "prompt_builder": {"query": query},
        }
    )

    category_result = json.loads(result["predictor_llm"]["replies"][0])
    # pprint(category_result)
    combined_result = {
        "url": url,
        "page_title": page_title,
        "domain": domain,
        "page_description": description_keywords.get("description") if description_keywords.get("description") else "",
        "page_keywords": description_keywords.get("keywords") if description_keywords.get("keywords") else [],
        "page_navbar_text": navbar_text,
        "predicted_title": category_result["title"],
        "predicted_category": category_result["category"],
        "predicted_description": category_result["description"],
    }
    return combined_result
