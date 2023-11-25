import os
from dotenv import load_dotenv
import camelot
import streamlit as st
#from PyPDF2 import PdfReader # Class for reading pdf
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS # Used when creating knowledge base
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI # Initialize the model from OpenAI
import openai

load_dotenv(dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("PDF Document Table Query App") # Upload PDF and start asking questions on it.
st.header("Ask Your PDF")


def path_to_pdf(file_path, uploaded_doc):
    with open(file_path, "wb") as file:
        file.write(uploaded_doc.read())

def extract_tables(file_path):
    try:
        return camelot.read_pdf(file_path, pages='all')
    except Exception as err:
        st.error(f"Error extracting tables: {str(err)}")
        return None

def display_tables(tables):
    st.subheader("Extracted Tables:")
    if tables:
        for table_num, table in enumerate(tables):
            st.write(f"Table {table_num + 1}")
            st.write(table.df)
    else:
        st.write("No tables found in the PDF")

#Step 1: Uploading PDF containing tables:
pdf = st.sidebar.file_uploader("Upload your PDF", type="pdf")

file_path = 'pdf_doc.pdf'

def main():
    # Check if the uploaded file is not none:
    if pdf is not None:
        st.info("Processing PDF... This may take a moment.")
        
        # Saving PDF to path:
        path_to_pdf('pdf_doc.pdf', pdf)
    
        # Step 2: PDF parsing and Table Extraction:
        tables = extract_tables(file_path)
        st.success("PDF file successfully uploaded!")

        # Display tables in PDF:
        display_tables(tables)
        # Selecting tables to Query:
        st.sidebar.subheader("Select Tables to Query:")
        selected_tables = st.sidebar.multiselect("Select a table to query:", range(1, len(tables) + 1))
        # Step 3: Geting user question or input
        user_question = st.text_input("Ask a question:")
        upload_button = st.button("Find Answer")

        if user_question or upload_button:
           # Step 4: Preprocessing table data (converting to string)
            table_text = ""
            for num, selected_table in enumerate(selected_tables):
                selected_table_df = tables[selected_table - 1].df

                # converting to string:
                table_text += f'\nTable {num + 1}: {selected_table_df.to_string(index=False)}'

            # Concatenate the user's question with the extracted text from the PDF for LLM:
            user_question_with_table = f"\n{table_text}\n\n{user_question}"

            # Step 5: Utilizing the LLM from GPT-3:
            # Setting up roles for the system and user:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant that can provide information from tables."},
                {"role": "user", "content": user_question_with_table}]
            
            # Use OpenAI GPT-3 to answer the question
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
            
            # Display the answer
            answer = response["choices"][0]["message"]["content"]
            st.success(answer)
       

if __name__ == "__main__":
    main()
