import streamlit as st
from main import run_pipeline
from tempfile import NamedTemporaryFile

st.title('PDF Processing App')

uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_pdf_path = tmp.name

    if st.button('Process PDF'):
        try:
            excel_path = run_pipeline(temp_pdf_path)

            st.success('PDF processed successfully.')
            with open(excel_path, "rb") as excel_file:
                st.download_button(label="Download Excel Output", data=excel_file,
                                   file_name="output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
