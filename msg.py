from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

messages =[
    SystemMessage(content='You are helpful asst'),
    HumanMessage(content='Tell me abt Langchain')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)