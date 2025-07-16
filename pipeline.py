import os
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

def load_and_split(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def build_qa_chain():
    pdf_dir = "docs"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    with ThreadPoolExecutor() as executor:
        docs = list(executor.map(load_and_split, pdf_files))

    all_docs = [doc for sublist in docs for doc in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm_model = ChatGroq(model="llama-3.3-70b-versatile")

    prompt_template = """
    You are answering as a helpful assistant.Donot preamble.Be clear and concise.

    Role: {role}
    Question: {input}
    Context: {context}

    Provide an answer specifically helpful for the role above.
    If you don't know the answer, just say that you don't know. Don't make anything up.
    """

    prompt = PromptTemplate(input_variables=["input", "context", "role"], template=prompt_template)
    document_chain = create_stuff_documents_chain(llm_model, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain
