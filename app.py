import os

import openai

from langchain.chat_models import AzureChatOpenAI

from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

from langchain.llms import AzureOpenAI

from langchain.utilities import BingSearchAPIWrapper

from langchain.agents.tools import Tool

from langchain import LLMMathChain

from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit

from langchain.agents import initialize_agent, AgentType




# Environment Variables

openai.api_base = os.getenv('OPENAI_API_BASE')

openai.api_key = os.getenv("OPENAI_API_KEY")




OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")

OPENAI_CHAT_ENGINE = os.getenv("OPENAI_CHAT_ENGINE")




OPENAI_COMPLETIONS_MODEL = os.getenv("OPENAI_COMPLETIONS_MODEL")

OPENAI_COMPLETIONS_ENGINE = os.getenv("OPENAI_COMPLETIONS_ENGINE")




# Constant values

openai.api_type = "azure"

openai.api_version = "2023-03-15-preview"




toolkit = AzureCognitiveServicesToolkit()




llm = AzureOpenAI(model_name=OPENAI_COMPLETIONS_MODEL, engine=OPENAI_COMPLETIONS_ENGINE, temperature=0.1)

model = AzureChatOpenAI(

    deployment_name=OPENAI_CHAT_MODEL,

    engine=OPENAI_CHAT_ENGINE,

    temperature=0

)





agent = initialize_agent(

    tools=toolkit.get_tools(),

    llm=llm,

    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,

    verbose=True,

)




agent.run(

    "What can I make with these ingredients?"

    "https://images.openai.com/blob/9ad5a2ab-041f-475f-ad6a-b51899c50182/ingredients.png"

)