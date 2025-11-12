import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

def get_azure_llm():
    """
    Returns a configured Azure OpenAI LLM client (LangChain compatible).
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")

    if not endpoint or not api_key:
        raise ValueError("Azure OpenAI credentials missing from environment variables.")

    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version="2024-12-01-preview",
        temperature=0,
        max_tokens=1500,
        verbose=True
    )
    return llm
