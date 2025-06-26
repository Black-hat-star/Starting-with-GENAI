from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Define the prompt template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chathistory'),
    ('human', '{query}')
])

# Load chat history
chat_history = []
with open('chathistory.txt') as f:
    for line in f:
        if line.startswith("HumanMessage"):
            content = line.split('content="', 1)[1].rsplit('")', 1)[0]
            chat_history.append(HumanMessage(content=content))
        elif line.startswith("AIMessage"):
            content = line.split('content="', 1)[1].rsplit('")', 1)[0]
            chat_history.append(AIMessage(content=content))

# Create prompt
prompt = chat_template.invoke({'chathistory': chat_history, 'query': 'where is my refund'})
print(prompt)
