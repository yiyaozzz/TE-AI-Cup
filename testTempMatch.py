from tempMatch_format_correction import process_pdf_to_landscape

def main():
    pdf_path="/Users/zyy/Desktop/TE/500000261553.pdf"
    output_dir="/Users/zyy/Desktop/TE/result"
    logo_template_path="/Users/zyy/Desktop/TE/latest/data/te_logo.png"
    
    # Call the function to process the PDF and orient images correctly based on the logo
    process_pdf_to_landscape(pdf_path, output_dir, logo_template_path)
    print("Processing complete. Check the output directory for the landscape images.")

if __name__ == "__main__":
    main()
