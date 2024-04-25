import streamlit as st  # Importing Streamlit for web app functionality
from PyPDF2 import PdfReader  # Importing PdfReader from PyPDF2 for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter for text chunking
import os  # Importing os for system operations
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Importing GoogleGenerativeAIEmbeddings for embeddings
import google.generativeai as genai  # Importing generativeai for Google AI configurations
from langchain.vectorstores import FAISS  # Importing FAISS for vector storage
from langchain_google_genai import ChatGoogleGenerativeAI  # Importing ChatGoogleGenerativeAI for conversational AI
from langchain.chains.question_answering import load_qa_chain  # Importing load_qa_chain for QA model loading
from langchain.prompts import PromptTemplate  # Importing PromptTemplate for prompts
from dotenv import load_dotenv  # Importing load_dotenv for environment variables


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def main():
    st.set_page_config("Chat PDF")
    st.header("Talk with PDF using Gemini")

    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF Processed!")

    generate_questions = st.checkbox("Generate Questions and Answers Automatically")

    if generate_questions and pdf_docs is not None:
        with st.spinner("Generating Questions..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search("dummy question")  # Using a dummy question for similarity search
            chain = get_conversational_chain()

            # Generate questions and answers
            questions = [
            "What is the purpose of The Economist's Big Mac Index?",
        "When was the Big Mac Index first introduced, and why?",
        "How does the Big Mac Index relate to the concept of purchasing power parity (PPP)?",
        "What factors contribute to the price of a Big Mac in different countries?",
        "Can you explain how the implied exchange rate is calculated using the Big Mac Index?",
        "What does it mean when a currency is considered overvalued or undervalued according to the Big Mac Index?",
        "Aside from the Big Mac Index, what are some other variants or similar indices mentioned in the text?",
        "What are some limitations of using the Big Mac Index as a measure of purchasing power parity?",
        "How was the Big Mac Index allegedly manipulated in Argentina, and what were the consequences?",
        "What are some of the variations in Big Mac prices and working time required to afford a Big Mac in different countries, as mentioned in the text? How does the geographical coverage of McDonald's franchises impact the applicability of the Big Mac Index?",
        "What are some factors mentioned in the text that can affect the price of a Big Mac in different countries?",
        "Why is the Big Mac Index sometimes criticized for its methodology?",
        "What is the significance of the variations in Big Mac prices among different countries, as mentioned in the text?",
        "Can you explain how the Big Mac Index reflects more than just relative currency values?",
        "What role do non-tradable goods and services play in understanding the limitations of the Big Mac Index?",
        "How does the Big Mac Index reflect differences in commercial strategies employed by McDonald's in various countries?",
        "What are some examples provided in the text that demonstrate the variability of Big Mac prices within a single country?",
        "How did the closure of McDonald's in Iceland impact the price of a Big Mac in that country?",
        "What are some statistics mentioned in the text regarding the most expensive and least expensive places to buy a Big Mac, as well as the average working time required to afford one in different cities?"
            ]
            answers = []
            for question in questions:
                answer = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                answers.append(answer["output_text"])

            st.write("Generated Questions:")
            for q, a in zip(questions, answers):
                st.write(f"Q: {q}\nA: {a}")


if __name__ == "__main__":
    main()
