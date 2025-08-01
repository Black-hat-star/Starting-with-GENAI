from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
load_dotenv()


model=GoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
res2=chain.invoke({'topic':'black hole'})

print(res2)

prompt = template.invoke({'topic': 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result)

print(final_result)
