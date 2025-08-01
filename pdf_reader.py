from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings # For Gemini Embeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Load the document
loader = TextLoader("docs.txt") # Ensure docs.txt exists
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings & store in FAISS
# This part would need to be adapted for GoogleGenerativeAIEmbeddings
# For example:
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever (fetches relevant documents)
retriever = vectorstore.as_retriever()

# Manually Retrieve Relevant Documents
# Manually Retrieve Relevant Documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# Combine Retrieved Text into a Single Prompt
# Ensure 'doc.page_content' is the correct attribute for your document objects
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize the LLM (using Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7) # Or your preferred Gemini model

# Manually Pass Retrieved Text to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.predict(prompt)

# Print the Answer
print("Answer:", answer)