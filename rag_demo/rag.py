import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 1. Load documents
# --------------------------------------------------
loader = TextLoader("data.txt")
documents = loader.load()

# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# --------------------------------------------------
# 3. Create embeddings + vector store
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --------------------------------------------------
# 4. Initialize LLM (Groq)
# --------------------------------------------------
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

# --------------------------------------------------
# 5. RAG prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

# --------------------------------------------------
# 6. RAG pipeline (manual but clear)
# --------------------------------------------------
def rag_query(question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        prompt.format_messages(
            context=context,
            question=question
        )
    )
    return response.content


# --------------------------------------------------
# 7. Test
# --------------------------------------------------
if __name__ == "__main__":
    print(rag_query("What is RAG?"))
