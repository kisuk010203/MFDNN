import PyPDF2


def merge_pdfs(pdf_list, output):
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output, "wb") as out:
        pdf_writer.write(out)


# List of pdfs to merge
pdfs = [f"Learning rate : {alpha}.pdf" for alpha in [0.01, 0.3, 4]]

# Output pdf file
output_pdf = "HW1_programming.pdf"

merge_pdfs(pdfs, output_pdf)
