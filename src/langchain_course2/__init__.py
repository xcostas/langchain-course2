import os
import subprocess
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field, PydanticOutputParser

# 1. Importaciones directas del estándar OpenTelemetry (A prueba de balas)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic_ai import Agent
#from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

windows_ip = subprocess.getoutput("ip route show | awk '/default/ {print $3}'").strip()
url_dinamica = f"http://{windows_ip}:11434"
nombre_modelo = os.getenv("OLLAMA_MODEL", "llama3.2")

# 2. Le decimos a OpenTelemetry cómo se llama nuestro proyecto
os.environ["PHOENIX_PROJECT_NAME"] = "mi-curso-langchain"

# 3. Creamos el puente "a mano", saltándonos el wrapper roto de Phoenix
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
provider.add_span_processor(SimpleSpanProcessor(exporter))

# 4. Conectamos LangChain al puente
LangChainInstrumentor().instrument(tracer_provider=provider)

class Source (BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field (description = "The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response with anwer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


def hello_world() -> None:
    
    
    # --- A partir de aquí, el código de tu curso ---

    
    llm = ChatOllama(
        temperature=0,
        model=nombre_modelo,
        base_url=url_dinamica
    ) 

    information = """
    Elon Reeve Musk (/ˈiːlɒn mʌsk/ Pretoria, 28 de junio de 1971) es un empresario, inversor... 
    (Pon aquí todo el texto de tu prompt)
    """

    symmary_template = """
    Dada la informacion {information} sobre una persona quiero que crees:
    1. Un resumen de la información proporcionada.
    2. Los logros más importantes de la persona.
    3. Las controversias o críticas relevantes.
    4. Cualquier otro dato relevante que se pueda extraer.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=symmary_template
    )

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})

    print(response.content)

#tavily = TavilyClient()

##@tool
##def search(query:str) -> str:
#    """
#    Toll hat searches over internet
#    Args:
#        query (str): The search query.
#    
#    Returns:
#        str: The search result.
#    """
#    print(f"Searching for {query}")
#    return tavily.search(query=query)

buscador_tavily = TavilySearch()


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.
    Args:
        query (str): The search query.
    """
    print(f"Llama 3.2 quiere buscar: {query}")
    # Le pasamos a Tavily estricta y únicamente el campo 'query', evitando que Llama meta fechas raras
    return buscador_tavily.invoke({"query": query})

llm = ChatOllama(
    temperature=0,
    model=nombre_modelo,
    base_url=url_dinamica
)
tools = [search]
#agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke ({"messages": [HumanMessage(content="What is the weather in Tokyo now")]} )
    print(result)


if __name__ == "__main__":  
    main()
