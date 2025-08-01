from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
import os

load_dotenv()

model= GoogleGenerativeAI(model="models/gemini-1.5-flash")
parser = StrOutputParser()

class feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description='give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=feedback)
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following text into poisitive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)


classifier_chain = prompt1| model | parser2
prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)


prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

#print(classifier_chain.invoke({'feedback':'this is a wondeful smarthphone'}))
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")  # default fallback — note: no condition here
)



chain = classifier_chain | branch_chain
print(chain.invoke({'feedback':'this is a terrible phone'}))