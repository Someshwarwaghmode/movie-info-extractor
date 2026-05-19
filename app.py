import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="🎬 Movie Info Extractor",
    page_icon="🎥",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #6366f1;
    padding: 10px;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
    padding: 12px;
}

.stButton button:hover {
    background: linear-gradient(90deg,#4f46e5,#7c3aed);
}

.result-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #374151;
    margin-top: 20px;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #6366f1;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎬 Movie Information Extractor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Extract important movie details using Mistral AI + LangChain</div>',
    unsafe_allow_html=True
)

# Text Area
# Text Area
movie_text = st.text_area(
    "Enter Movie Paragraph",
    height=220,
    placeholder="Paste movie paragraph here..."
)

# Example Data
st.markdown("### 🎬 Example Movie Paragraph")

example_text = """
3 Idiots is a 2009 Indian Hindi-language comedy-drama film directed by Rajkumar Hirani.
It stars Aamir Khan, R. Madhavan, Sharman Joshi, Kareena Kapoor, and Boman Irani.

The story follows three engineering students who navigate friendship, academic pressure, and self-discovery at a prestigious college.

The movie was released on 25 December 2009 and became one of the highest-grossing Indian films.
It received praise for its humor, emotional depth, and message about creativity and education.
"""

st.code(example_text, language="text")

# Button
analyze_btn = st.button("🚀 Analyze Movie")

# LLM Setup
llm = init_chat_model(
    "open-mistral-7b",
    model_provider="mistralai",
)

# Prompt Template
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

# Chain
chain = movie_prompt | llm

# Analyze
if analyze_btn:

    if movie_text.strip() == "":
        st.warning("⚠ Please enter movie paragraph.")
    else:
        with st.spinner("Analyzing Movie Information... 🎥"):
            try:
                response = chain.invoke(
                    {
                        "movie_text": movie_text
                    }
                )

                st.markdown(
                    f"""
                    <div class="result-box">
                    <h3>🎯 Extracted Movie Information</h3>
                    <p>{response.content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error: {e}")