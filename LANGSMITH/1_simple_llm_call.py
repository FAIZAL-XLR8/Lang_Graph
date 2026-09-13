from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatHuggingFace(
    llm = (
        HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct', 
        task='text-generation')
    )
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
