from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#config contains tags,metadata as a form of dictionary
#tags - to label the run
#metadata - to provide additional information (optional)
load_dotenv()
import os
os.environ['LANGCHAIN_PROJECT'] = 'sequential_chain'

model = ChatHuggingFace(
    llm = (
        HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct', 
        task='text-generation')
    )
)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)


parser = StrOutputParser()
config={
    'tags':['sequential_chain', 'huggingface'],
    'metadata':{'description' : 'Sequential chain example'},
    'run_name' : 'Sequential_Chain3'
}
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Thor, the thunder god from Ragnorok anime'}, config=config)

print(result)
