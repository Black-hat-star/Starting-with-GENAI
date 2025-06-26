from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

Chat_template= ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms,what is {topic}')

])

prompt = Chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)