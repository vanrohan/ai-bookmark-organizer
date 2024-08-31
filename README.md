This is a companion repository to the blog post [Using AI to declutter and organise my bookmarks](https://vanderwalt.de/blog/ai-bookmark-organizer) 

To install dependencies:
    `poetry install`

To start the headless Chromium docker container:
    `docker compose -f docker-compose.yml up --detach=true`

To run: 
    `poetry run python main.py -f bookmarks_30_08_2024.html --chrome localhost:9223`
