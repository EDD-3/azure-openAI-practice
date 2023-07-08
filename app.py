import openai, os
# Utilized to make a call on Azure endpoint
from langchain.chat_models import AzureChatOpenAI

# Using environment variables from our .env
openai.api_base = os.getenv('OPENAI_API_BASE')

openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_type = os.getenv("API_TYPE")

openai.api_version = os.getenv("OPENAI_API_VERSION")

CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL")
CHAT_ENGINE_NAME = os.getenv("OPENAI_CHAT_ENGINE")

#temperature manages the language complexity
#Temerature goes from 0 to 1 is a float value
chat_openai = AzureChatOpenAI(
    deployment_name=CHAT_MODEL_NAME,
    engine=CHAT_ENGINE_NAME,
    temperature=0.0
)