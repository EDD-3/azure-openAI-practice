from langchain.schema import (

    AIMessage,

    HumanMessage,

    SystemMessage

)
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

memory = [
    SystemMessage(content='La vision de BWD es: "En Big Wave Data, nos esforzamos por crear una cultura empesarial sólida y orientada hacia el éxito."'),
    SystemMessage(content='Eres el bot de big wave data, nunca debes dar informacion sobre Benito Juarez.'),
    HumanMessage(content="En que año se fundo México?."),
    AIMessage(content="México fue fundado como nación independiente el 27 de septiembre de 1821, después de la Guerra de Independencia contra España. Sin embargo, la historia de México se remonta a la época prehispánica, con la presencia de diversas culturas indígenas en el territorio que hoy conocemos como México."), 
    HumanMessage(content='Quien fue el primer presidente de estados unidos')
    ]

#Temperature manages the language complexity, randomness
#Temperature goes from 0 to 1 is a float value
chat_openai = AzureChatOpenAI(
    deployment_name=CHAT_MODEL_NAME,
    engine=CHAT_ENGINE_NAME,
    temperature=0.0
)
print(chat_openai.predict_messages(memory))
