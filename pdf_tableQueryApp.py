import os
import camelot
import streamlit as st
import pandas as pd
from io import StringIO

from transformers import TapasForQuestionAnswering, AutoTokenizer, TapasTokenizer, TFBartForConditionalGeneration, pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer


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

def display_tables(tables, selected_tables1):
    if selected_tables1 is not None:
        for num in selected_tables1:
            #if num in range(len(tables)):
            st.write(f"Table {int(num)}")
            st.write(tables[int(num) - 1].df)
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
        st.success(f"I found {len(tables)} tables in your file. Please select the table(s) to query")

        # Selecting tables to Query:
        st.sidebar.subheader("Select Tables to Query:")
        selected_tables = st.sidebar.multiselect("Select a table to query:", range(1, len(tables) + 1))

        # Display tables in PDF:
        st.sidebar.subheader("Display Selected Table(s):")
        selected_tables1 = st.sidebar.multiselect("Select a table to display:", range(1, len(tables) + 1))

        if selected_tables1:
            st.subheader("Selected Table(s):")
            display_tables(tables,selected_tables1)

        # Step 3: Geting user question or input
        user_question = st.text_input("Ask a question:")
        upload_button = st.button("Find Answer")

        if user_question or upload_button:
        # Step 4: Preprocessing table data (converting to string)
            table_num = selected_tables[0]
            df = tables[table_num-1].df
            df_str = df.to_string(index=False, header=False)

            # Loading the Flan T5 model and tokenizer:
            model_name = 'google/flan-t5-base'
            t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

            input_text = f"question: {user_question} context: {df_str}"
            inputs = t5_tokenizer.encode(input_text, return_tensors="pt")
            outputs = t5_model.generate(inputs)
            answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


            st.success(f"Answer: {answer}")

if __name__ == "__main__":
    main()
