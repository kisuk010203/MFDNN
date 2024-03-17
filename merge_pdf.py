import PyPDF2
import os
import sys


def merge_pdfs(pdf_list, output):
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output, "wb") as out:
        pdf_writer.write(out)


def merge_pdfs_from_directory(directory):
    assignment_number = directory.split("/")[-1][2:]
    pdf_files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".pdf") and not "hw" in filename
    ]
    pdf_files.sort()
    merge_pdfs(pdf_files, f"{directory}/HW{assignment_number}_programming.pdf")


merge_pdfs_from_directory(sys.argv[1])
