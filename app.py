import streamlit as st
import subprocess
import json
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

st.title('PDF Processing App')

uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
if uploaded_file is not None:
    with TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "uploaded_file.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        process = subprocess.run(
            ['python', 'full_pipeline_processor.py', pdf_path, temp_dir], capture_output=True, text=True)

        if process.returncode == 0:
            excel_path = os.path.join(temp_dir, "output.xlsx")
            with open(excel_path, "rb") as excel_file:
                st.download_button(label="Download Excel Output", data=excel_file,
                                   file_name="output.xlsx", mime="application/vnd.ms-excel")
        else:
            st.error("Failed to process PDF")
