from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file

def get_openrouter_with_groq(model: str = "meta-llama/llama-3.3-70b-instruct") -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model,
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base=getenv("OPENROUTER_BASE_URL"),  # e.g. "https://openrouter.ai/api/v1"
        # Pass the provider preference as an extra parameter
        # model_kwargs={"order": ["Nebius AI Studio"]}
        # model_kwargs={"provider": {"order": ["Groq"]}}
        extra_body={
            "provider": {
                "order": ["Nebius"],
                "allow_fallbacks" : True,

            }
        }
    )

# Instantiate your LLM with the Groq provider preference
llm = get_openrouter_with_groq()

# Use your LLM as usual; every call will now include the provider preference
response = llm.invoke("Tell me a fun fact about space.")
print(response)