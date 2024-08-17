import streamlit as st
from pdf_processing import extract_text_from_pdf
from retriever import create_index, retrieve_documents, generate_answer


st.title("RAG System for PDF Question Answering")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file is not None:
    text = extract_text_from_pdf(pdf_file)
    index, sentences = create_index(text)
    st.write("PDF successfully processed!")

    query = st.text_input("Ask a question about the PDF")
    if query:
        retrieved_docs = retrieve_documents(query, index, sentences)
        context = " ".join(retrieved_docs)
        answer = generate_answer(context, query)
        st.write(f"Answer: {answer}")
