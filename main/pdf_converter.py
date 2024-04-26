import fitz  
import os


def convert_pdf_to_png(pdf_path, dpi=250):
    doc = fitz.open(pdf_path)
    output_dir = f"{os.path.splitext(pdf_path)[0]}_pages"
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    num_pages_to_process = min(doc.page_count, 17)

    for page_num in range(num_pages_to_process):
        if page_num in [1, 2, 3]:
            continue
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        output_file = os.path.join(output_dir, f"page_{page_num+1}.png")

        pix.save(output_file)
        image_paths.append(output_file)

    doc.close()

    return image_paths
