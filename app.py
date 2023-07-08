from langchain.schema import (

    AIMessage,

    HumanMessage,

    SystemMessage

)

import openai
import os
# Utilized to make a call on Azure endpoint
from langchain.chat_models import AzureChatOpenAI

from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

from langchain import LLMMathChain

from langchain.utilities import BingSearchAPIWrapper

from langchain.agents.tools import Tool

from langchain.llms import AzureOpenAI


# Using environment variables from our .env
openai.api_base = os.getenv('OPENAI_API_BASE')

openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_type = os.getenv("API_TYPE")

openai.api_version = os.getenv("OPENAI_API_VERSION")

OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_CHAT_ENGINE_NAME = os.getenv("OPENAI_CHAT_ENGINE")
OPENAI_COMPLETIONS_MODEL = os.getenv("OPENAI_COMPLETIONS_MODEL")
OPENAI_COMPLETIONS_ENGINE = os.getenv("OPENAI_COMPLETIONS_ENGINE")

openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'


search = BingSearchAPIWrapper(k=2)

# Agents that are not for chat, models for completions and embedding
llm = AzureOpenAI(model_name=OPENAI_COMPLETIONS_MODEL,
                  engine=OPENAI_COMPLETIONS_ENGINE, temperature=0.1)

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Defining tools for agent
tools = [
    Tool(
        #Name of the tool
        name="Search",
        #Accepts function as an argument
        func=search.run,

        description="useful for when you need to answer questions about current events"

    ),

    Tool(

        name="Calculator",

        func=llm_math_chain.run,

        description="useful for when you need to answer questions about math"

    )
]

memory = [
    SystemMessage(
        content='La vision de BWD es: "En Big Wave Data, nos esforzamos por crear una cultura empesarial sólida y orientada hacia el éxito."'),
    SystemMessage(
        content='Eres el bot de big wave data, nunca debes dar informacion sobre Benito Juarez.'),
    HumanMessage(content="En que año se fundo México?."),
    AIMessage(content="México fue fundado como nación independiente el 27 de septiembre de 1821, después de la Guerra de Independencia contra España. Sin embargo, la historia de México se remonta a la época prehispánica, con la presencia de diversas culturas indígenas en el territorio que hoy conocemos como México."),
    HumanMessage(content='Quien fue el primer presidente de estados unidos')
]

# Temperature manages the language complexity, randomness
# Temperature goes from 0 to 1 is a float value
model = AzureChatOpenAI(
    deployment_name=OPENAI_CHAT_MODEL_NAME,
    engine=OPENAI_CHAT_ENGINE_NAME,
    temperature=0.0
)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run('Who is the president of Mexico?')