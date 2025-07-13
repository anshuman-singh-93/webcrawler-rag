import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

# Recursively get all links from a website
def get_all_links_recursive(url, domain=None, visited=None):
    if visited is None:
        visited = set()
    if domain is None:
        domain = urlparse(url).netloc
    if url in visited:
        return visited
    visited.add(url)
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            link = urljoin(url, a_tag['href'])
            parsed_link = urlparse(link)
            if parsed_link.netloc == domain and link not in visited:
                get_all_links_recursive(link, domain, visited)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return visited


# Step 1: Crawl website
start_url = "https://anshumansingh.me"  # Replace with your website
all_links = get_all_links_recursive(start_url)
all_links_list = list(all_links)

# Step 2: Load documents
loader = WebBaseLoader(web_path=all_links_list)
docs = loader.load()

# Step 3: Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
final_docs = splitter.split_documents(docs)

# Step 4: Create vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents=final_docs, embedding=embeddings)

# Step 5: LLM setup and retrieval chain
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template(
    """
Answer the  question based only on the provided context:

question:{input}

<context>
{context}
</context>
"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 6: Query the chain
response = retrieval_chain.invoke({"input": "how many research papers are found?"})