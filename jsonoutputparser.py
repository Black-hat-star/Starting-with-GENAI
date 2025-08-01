from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()


model=GoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)
parser=JsonOutputParser()

template = PromptTemplate(
    template='Give me the name,age,city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)

#print(result)

#finalresult = parser.parse(result)

#print(finalresult)

#print(type(finalresult))

chain = template | model | parser
result = chain.invoke({})

print(result)
print(type(result))