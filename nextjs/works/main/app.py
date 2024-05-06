import streamlit as st
import os
import sys
from main import process_pdf_or_folder


def is_pdf_file(file):
    return file.name.lower().endswith('.pdf')


st.title('PDF Processor')

uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
uploaded_folder = st.text_input(
    "Or specify a path to a directory containing PDF files:")

if uploaded_file:
    if is_pdf_file(uploaded_file):
        with st.spinner("Processing PDF..."):
            output_message = process_pdf_or_folder(uploaded_file, is_file=True)
            st.success(output_message)
    else:
        st.error("Please upload a valid PDF file.")

elif uploaded_folder:
    if os.path.isdir(uploaded_folder):
        with st.spinner("Processing all PDFs in folder..."):
            output_message = process_pdf_or_folder(
                uploaded_folder, is_file=False)
            st.success(output_message)
    else:
        st.error("The specified path is not a valid directory.")
