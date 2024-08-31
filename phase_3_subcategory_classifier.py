import json
import logging
import os
from pprint import pprint
from typing import Any, Dict

from dotenv import find_dotenv, load_dotenv
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

# from WebsiteFetcher import WbsiteFetcher

load_dotenv(find_dotenv())
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.__getattribute__(str(os.getenv("LOG_LEVEL", "INFO"))))


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


async def run_pipeline_phase_3(original_predictions, ollama_host, ollama_model):
    with open("prompts/system_prompt.jinja") as template_file:
        system_prompt = template_file.read()
    query = """ Your task is to create a sub-category for a website bookmark within the parent category. 
                The new sub-category should still be high-level in order to accommodate future bookmarks.

                The parent category is "%s".
                The bookmark Title is (%s). 
                The bookmark Description is (%s).

                Based on this information, answer the following questions:
                1. What is a good Sub-Category for this website?
                    - IMPORTANT - This must not be similar to the main category.
                    - Be short and concise (2 words maximum, preferably 1 word)
                    - Do not use vague words like "Home", "Page" or "Startseite".
                    - Do not use irrelevant numbers.
                    - Use proper Capitalization
                    - Avoid numbers unless necessary for clarity.
                    - Use & instead of and, and other shortforms.

                Output your answers as a JSON object with the following structure:
                {
                    "sub_category": str,
                }
                """ % (
        original_predictions["predicted_category"],
        original_predictions["predicted_title"],
        original_predictions["predicted_description"],
    )

    predictor_generator = MyOllamaGenerator(
        model=ollama_model,
        url=f"{ollama_host}/api/generate",
        generation_kwargs={
            "num_predict": 550,
            "temperature": 0.75,
        },
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=system_prompt))
    pipe.add_component("predictor_llm", predictor_generator)

    pipe.connect("prompt_builder", "predictor_llm")

    logger.debug(f"Starting pipeline for {original_predictions['url']}")
    result = pipe.run(
        {
            "prompt_builder": {"query": query},
        }
    )
    try:
        category_result = json.loads(result["predictor_llm"]["replies"][0])
    except json.JSONDecodeError:
        logger.exception(f"Failed to decode JSON from Ollama response for {original_predictions['url']}")
        category_result = {"sub_category": None}
        pprint(result)

    original_predictions["predicted_sub_category"] = category_result["sub_category"]
    return original_predictions
