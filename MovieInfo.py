from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from rich import print
from dotenv import load_dotenv 
load_dotenv()

llm = init_chat_model(
    "open-mistral-7b",
    model_provider="mistralai",
)

movie_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """
You are an expert AI assistant for movie information extraction.

Your job is to carefully analyze movie-related paragraphs and identify all important movie details.

Extract as much useful information as possible from the text, including:

- title
- release_year
- genre
- director
- main_cast
- language
- imdb_rating
- storyline
- sentiment
- summary

Rules:
- Extract only information present in the paragraph.
- Do not guess missing information.
- Keep the response clean and readable.
- If multiple movies are present, extract information for each movie separately.
- Provide complete and detailed movie understanding from the paragraph.
"""
    ),
    (
        "human",
                """
        Analyze the following movie paragraph and extract all important movie information:

        {movie_text}
        """
    )
    ]
)

movie_paragraph = """
3 Idiots is a 2009 Indian Hindi-language comedy-drama film directed by Rajkumar Hirani.
It stars Aamir Khan, R. Madhavan, and Sharman Joshi.
The movie was released on 25 December 2009 and became one of the highest-grossing Indian films.
"""

chain = movie_prompt | llm

response = chain.invoke(
    {
        "movie_text": movie_paragraph
    }
)

print(response)