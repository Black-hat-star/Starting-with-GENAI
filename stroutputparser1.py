from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


model=GoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

#1st prompt
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


#2nd prompt
template2=PromptTemplate(
    template='Write a 5line summary on the following text ./n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result})

result1= model.invoke(prompt2)

print(result1)


parser=StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)