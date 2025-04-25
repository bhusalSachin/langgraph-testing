from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# LLM model for all the calls
llm = ChatOpenAI(model="gpt-4o-mini")