from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']  # must match variable used in the template
)

model = GoogleGenerativeAI(model="models/gemini-1.5-flash")

parser = StrOutputParser()
chain = prompt | model | parser

result = chain.invoke({'topic': 'cricket'})

print(result)
chain.get_graph().print_ascii()
