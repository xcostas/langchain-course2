import os
import subprocess
from dotenv import load_dotenv

# 1. Importaciones directas del estándar OpenTelemetry (A prueba de balas)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main() -> None:
    # 2. Le decimos a OpenTelemetry cómo se llama nuestro proyecto
    os.environ["PHOENIX_PROJECT_NAME"] = "mi-curso-langchain"
    
    # 3. Creamos el puente "a mano", saltándonos el wrapper roto de Phoenix
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    
    # 4. Conectamos LangChain al puente
    LangChainInstrumentor().instrument(tracer_provider=provider)
    
    # --- A partir de aquí, el código de tu curso ---
    windows_ip = subprocess.getoutput("ip route show | awk '/default/ {print $3}'").strip()
    url_dinamica = f"http://{windows_ip}:11434"
    nombre_modelo = os.getenv("OLLAMA_MODEL", "phi3")
    
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

if __name__ == "__main__":
    main()