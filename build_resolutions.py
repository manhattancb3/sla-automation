'''


python-docx - manipulate word files
docxtpl - template filler (MAY HAVE DEPRECATED DEPENDENCIES SOON)
pandas
python-docx
openpyxl
pywin32



- template should hold styles and formatting
- Python should only insert data

STEPS:
- build template doc with replaceable fields
- fill template fields from correct row/columns in spreadsheet
- check for any potential issues, flag for user
- convert to PDF if no issues detected
- save .docx and .pdf files into new output folder

'''

from docxtpl import DocxTemplate
import pandas as pd
import os
import datetime as dt

parent_folder = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space"
template_path = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\test_template.docx"
spreadsheet_path = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\test_spreadsheet.csv"

# _____________ MINI EXAMPLE / TEST ________________ #
# Load the template
template = DocxTemplate(template_path)

# Read the CSV spreadsheet
df = pd.read_csv(spreadsheet_path)

# Loop over rows
for idx, row in df.iterrows():
    print("idx", idx)
    print("row", row)
    context = {
        "street_address": row["street_address"],
        "first_name": row["first_name"]
    }
    # Render the template with this row's data
    template.render(context)

    # Save template to new document
    template.save(os.path.join(parent_folder, f"output_{idx+1}.docx"))

print("Documents generated.")

# _____________________________________________________ #

# __________________ FULL IMPLEMENTATION ________________ #

parent_folder = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space"
template_path = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\test_template.docx"
spreadsheet_path = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\test_spreadsheet.csv"
output_folder = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\Resolution Doc Outputs"

# Read the CSV spreadsheet
df = pd.read_csv(spreadsheet_path)

# Loop over rows
for idx, row in df.iterrows():
    # Load the template
    template = DocxTemplate(template_path)
    
    context = {
        "today_date": dt.now().strftime("%B %d, %Y"),
        "applicant_name": row["applicant_name"],
        "street_address": row["street_address"],
        "zip_code": "10003",    # DEFAULTED TEMPORARILY,
        "month": dt.now().strftime("%B"),
        "year": dt.now().year, 
        "first_name": row["first_name"]
    }
    # Render the template with this row's data
    template.render(context)

    # Save template to new document
    template.save(os.path.join(parent_folder, f"output_{idx+1}.docx"))