from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt1=PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = GoogleGenerativeAI(model="models/gemini-1.5-flash")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'unemployemenet in india'})
print(result)
chain.get_graph().print_ascii()