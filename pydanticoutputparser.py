from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
load_dotenv()

model=GoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n{format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
    

chain = template | model | parser
final_res=chain.invoke({'place':'british'})
print(final_res)

prompt = template.invoke({'place': 'indian'})
print(prompt)
result = model.invoke(prompt)

final_result = parser.parse(result)

print(final_result)