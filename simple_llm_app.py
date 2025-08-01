from langchain.llms import googlegenerativeai
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = googlegenerativeai(model_name="models/gemini-1.5-flash", temperature=0.7)

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Define the input
topic = input('Enter a topic: ')

# Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call the LLM directly
blog_title = llm.predict(formatted_prompt)

# Print the output
print("Generated Blog Title:", blog_title)