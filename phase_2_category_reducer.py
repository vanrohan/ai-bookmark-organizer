import json
import logging
import os
from pprint import pprint
from typing import Any, Dict

from dotenv import find_dotenv, load_dotenv
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

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


async def run_pipeline_phase_2(bookmarks_by_category, ollama_host, ollama_model):
    with open("prompts/system_prompt.jinja") as template_file:
        system_prompt = template_file.read()  # Read the whole file as string into variable

    query = """Your task is to group similar/overlapping categories together based on their content similarity. 
                Less categories are better, and do not create more than 16 final categories.

                This is the list of categories that need grouping:
                %s.

                For the new grouped categories:
                    - Take care not to forget any input categories.
                    - Always use the concisest category name for the new name.
                    - Be short and concise (2 words maximum, preferably 1 word)
                    - Do not use irrelevant numbers.
                    - Use proper Capitalization and spacing
                    - Avoid special characters or numbers unless necessary for clarity.
                    - Use & instead of and, and other shortforms.

                Generate a JSON object that represents these groupings, where each key is a category name and value is an array of similar categories.
                    { [category: string]: string[] }
            """ % (
        ", ".join(
            [
                k
                for k in bookmarks_by_category.keys()
                if k
                not in [
                    "Dead",
                ]
            ]
        )
    )

    logger.info(f"Connecting to {ollama_host} using model {ollama_model}")
    predictor_generator = MyOllamaGenerator(
        model=ollama_model,
        url=f"{ollama_host}/api/generate",
        generation_kwargs={
            "num_predict": 1500,
            "temperature": 0.75,
        },
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=system_prompt))
    pipe.add_component("predictor_llm", predictor_generator)

    pipe.connect("prompt_builder", "predictor_llm")

    logger.debug("Starting pipeline for category grouping")
    result = pipe.run(
        {
            "prompt_builder": {"query": query},
        }
    )

    try:
        category_result = json.loads(result["predictor_llm"]["replies"][0])
    except Exception:
        pprint(result)
        logger.exception("Error parsing JSON from Ollama")

    # pprint(category_result)
    result = bookmarks_by_category
    for key, value in category_result.items():
        if key not in result:
            # New category needs to be added
            result[key] = []
        # Moving all old category contents into the new category
        for old_category in value:
            if old_category == key:
                continue
            if old_category in result:
                result[key].extend(result[old_category])
                if old_category not in category_result.keys():
                    del result[old_category]
    return result
