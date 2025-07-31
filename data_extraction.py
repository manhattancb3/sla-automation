'''

List of pip-installs
- matplotlib


pytesseract (requires installation)
- Information here: https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file
- Windows installer here: https://github.com/UB-Mannheim/tesseract/wiki

poppler (requires installation)
- Info here: https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows
- Download from github page: https://github.com/oschwartz10612/poppler-windows/releases
- Add to PATH per stackoverflow link above


NEEDED UPDATES?
- Add confidence score checks for OCR. This will compare unprocessed OCR against
    preprocessed OCR and use the one with the higher confidence score.
- automatic generation of file paths
    - for example, the application PDFs and Vote Sheet .docx. Ideally these are placed
        in a single folder and then this script finds them and saves their paths.
- warnings if extracted data is unusual
    - certain checks can be added depending on the info
    - for example, names shouldn't have any symbols besides hyphens
    - resolutions should probably have a relatively standard character limit
- Make "Applicant or Licensee Name" and "Trade Name" interchangeable when searching
    for resolution in vote sheet?
- create "clean" names

'''

import os
from pdf2image import convert_from_path
from PIL import ImageOps
import re
import pandas as pd
import pytesseract
from docx import Document

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ______________ RUNTIME TRACKING - START _________________________ #
# Used to measure and output program runtime
import time
start_time = time.time()
# _________________________________________________________________ #

# Specifies location of tesseract executable if not set in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Removes horizontal and vertical lines to improve OCR
def remove_form_lines(image, debug=False):
    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt


    # Invert image to make lines white
    invert = cv2.bitwise_not(image)
    thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Estimate kernel sizes based on image width/height
    img_height, img_width = thresh.shape

    # # Horizontal line removal
    h_kernel_len = img_width // 30  # Adjust divisor based on form layout
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 10))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

    if debug:
        # cv2.imshow("Horizontal Lines", detect_horizontal)
        # cv2.waitKey(0)
        plt.imshow(detect_horizontal, cmap='gray')
        plt.show()

    # Vertical line removal
    v_kernel_len = img_height // 60
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, v_kernel_len))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    contours = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        cv2.drawContours(image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

    if debug:
        # cv2.imshow("Vertical Lines", detect_vertical)
        # cv2.waitKey(0)
        plt.imshow(detect_vertical, cmap='gray')
        plt.show()

    return image

# Removes skew from image to improve OCR
def deskew_image_hough(cleaned, canny_thresh1=50, canny_thresh2=150, hough_threshold=200, min_angle=1.0):
    """
    Deskews an image using Hough Line Transform to detect dominant angles.

    Parameters:
        cleaned (numpy.ndarray): Input binary image.
        canny_thresh1 (int): Lower threshold for Canny edge detection.
        canny_thresh2 (int): Upper threshold for Canny edge detection.
        hough_threshold (int): Minimum votes to detect a line in Hough Transform.
        min_angle (float): Minimum angle in degrees to apply correction.

    Returns:
        deskewed (numpy.ndarray): The deskewed image.
        median_angle (float): The detected skew angle.
    """
    # Step 1: Edge detection
    edges = cv2.Canny(cleaned, canny_thresh1, canny_thresh2, apertureSize=3)

    # Step 2: Hough Line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90  # Convert to degrees relative to horizontal

            # Focus on near-horizontal lines only
            if abs(angle) < 45:
                angles.append(angle)

    if angles:
        median_angle = np.median(angles)
    else:
        median_angle = 0.0

    # Step 3: Deskew if angle is significant
    if abs(median_angle) > min_angle:
        (h, w) = cleaned.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(cleaned, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = cleaned

    return deskewed, median_angle

# Performs multiple preprocessing steps to facilitate OCR
def preprocess_for_ocr(jpeg_image):
    '''
    Applies various preprocessing strategies for improving OCR success. For this application,
    the OCR is working with PDF files that were made from a scanned printout which often 
    involve a lot of visual noise, low contrast, etc.

    Potential enhancements:

    '''
    import cv2
    import numpy as np
    from PIL import ImageFilter, ImageOps, Image, ImageEnhance

    # 1 Grayscale
    gray = jpeg_image.convert('L')

    # 2 Resize (2x better than 3x)
    gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)

    # 3 Contrast enhancement
    enhancer = ImageEnhance.Contrast(gray)
    contrasted = enhancer.enhance(1.5)

    # 4 Sharpen
    sharpened = contrasted.filter(ImageFilter.SHARPEN)

    # 5 Convert to OpenCV format
    open_cv_image = np.array(sharpened)
    # cleaned = open_cv_image

    # deskewed_image, detected_angle = deskew_image_hough(open_cv_image)
    # print(f"Detected skew angle: {detected_angle} degrees")

    # 6 Remove horizontal and vertical boxes/lines
    cleaned = remove_form_lines(open_cv_image, debug=False)

    # # Invert for easier contour detection
    # invert = cv2.bitwise_not(cleaned)
    # thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # # Remove horizontal lines
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    # detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # for c in contours:
    #     cv2.drawContours(cleaned, [c], -1, (255, 255, 255), 2)

    # # Remove vertical lines
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    # detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    # contours = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # for c in contours:
    #     cv2.drawContours(cleaned, [c], -1, (255, 255, 255), 2)


    # 6 Thresholding (MAKES OCR WORSE!!!!!!!!)
    #thresh = cv2.adaptiveThreshold(open_cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 7 Morphological cleaning
    # kernel = np.ones((1, 1), np.uint8)
    # cleaned = cv2.morphologyEx(open_cv_image, cv2.MORPH_OPEN, kernel)

    # 8 Deskew
    # coords = np.column_stack(np.where(cleaned > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # print(angle)

    # # Normalize angle
    # if angle < -45:
    #     angle = 90 + angle
    # elif angle > 45:
    #     angle = angle - 90

    # # Ignore small angles (optional)
    # if abs(angle) > .25:  # degrees
    #     print("------------------descewed run-----------------")
    #     (h, w) = cleaned.shape[:2]
    #     center = (w // 2, h // 2)
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     deskewed = cv2.warpAffine(cleaned, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # else:
    #     deskewed = cleaned  # No correction needed

    # 9 Denoise
    # denoised = cv2.medianBlur(deskewed, 3) # faster processing option
    denoised = cv2.fastNlMeansDenoising(cleaned, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert back to PIL
    return Image.fromarray(denoised)

# Extracts needed information from .txt files generated from OCR
def extract_info(text, path):
    '''
    Extracts specific information from the OCR text outputs. The OCR converts the PDFs
    into plain text which facilitates the data extraction using regular expressions.

    Needed updates:
    - add warnings if name is unusual (too long, contains symbols, etc.)
    - remove characters that are artifacts of text field boxes (|, [, etc.)
    '''

    info = {}   # dict to store extracted data

    # Extract Applicant
    applicant_match = re.search(
        r'Applicant or Licensee Name[:;\]\|\\/ \t]*([A-Z][^\n]*)',
        # r'Applicant or Licensee Name[:;\]\|\s]*([A-Z][^\n]*?)(?=\n\s*\d+\.)',
        # r'Applicant or Licensee Name[:;\]\|\s]*([A-Z][\w\s\.,&\'-]+)',
        text,
        re.IGNORECASE
    )
    if applicant_match:
        info['applicant_name'] = applicant_match.group(1).strip()
    else:
        info['applicant_name'] = f'=HYPERLINK("{path}", "!! --- REQUIRES MANUAL ENTRY --- !!")'


    # Extract Street Address (until a line break or next label)
    street_address_match = re.search(
        r'Street Address of Establishment[:;\]\|\\/ \t]*([A-Z0-9][^\n]*)',
        # r'Street Address of Establishment[:;\]\|\s]*([A-Z0-9][^\n]*?)(?=\n\s*\d+\.)',
        # r'Street Address of Establishment[:;\]\|\s]*([A-Z0-9][\w\s\.,#&\'/-]+)', 
        text,
        re.IGNORECASE)
    if street_address_match:
        info['street_address'] = street_address_match.group(1).strip()
    else:
        info['street_address'] = f'=HYPERLINK("{path}", "!! --- REQUIRES MANUAL ENTRY --- !!")'
    

    # Extract Representative Name(s)
    # Find relevant field first
    rep_name_field_match = re.search(
        r"Representative/Attorney's Full\s?Name[:;\]\|\\/ \t]*\[?(.+)",
        # r"Representative/Attorney's Full\s?Name[:;\]\|\s]*\[?(.+)",
        text,
        re.IGNORECASE
    )
    if rep_name_field_match:
        field_text = rep_name_field_match.group(1)
        # Split off at law firm indicators or common break points
        cleaned_field = re.split(r'\s+(?:c/o|–|-|,|;|:)\s+', field_text)[0]
        # Extract possible person names with optional middle initials and suffixes
        rep_names = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+(?:,\s*(?:Esq\.|Jr\.|Sr\.|II|III))?)\b',
            cleaned_field
        )
        if rep_names:
            info['representative_name(s)'] = rep_names
        else:
            info['representative_name(s)'] = f'=HYPERLINK("{path}", "!! --- REQUIRES MANUAL ENTRY --- !!")'
    else:
        info['representative_name(s)'] = f'=HYPERLINK("{path}", "!! --- REQUIRES MANUAL ENTRY --- !!")'


    # Extract Representative Email(s)
    # Find relevant field first
    rep_email_field_match = re.search(
        r'Business\s+E-?mail\s+Address.*?:\s*(.+)',
        text,
        re.IGNORECASE
    )
    # Find email addresses within relevant field text
    if rep_email_field_match:
        field_text = rep_email_field_match.group(1)
        emails = re.findall(
            r'[\w\.-]+@[\w\.-]+\.\w+',
            field_text
        )
        if emails:
            info['representative_email(s)'] = emails
        else:
            info['representative_email(s)'] = f'=HYPERLINK("{path}", "!! --- REQUIRES MANUAL ENTRY --- !!")'
    else:
        info['representative_email(s)'] = f'=HYPERLINK("{path}", "!! --- REQUIRES MANUAL ENTRY --- !!")'


    return info

# Searches for and provides specific resolution text from Minutes document
def find_reso(document, name, end_name):
    '''
    Searches Vote Sheet .docx file for resolution text for each SLA applicant.

    Returns a string of text containing the resolution for each SLA application.

    Needed updates:
    - warning for missing info + link to file

    notes:
    - vote sheets are not in a consistent order
    - end of reso found with these string checks (in order of precedence):
        - next applicant name = end match without this line
        - "not heard at committee" IF after SLA section header = end match without this line
        - "new liquor license application" = end match without this line
        - "alterations" = end match without this line
            - needs more specific text match
        - "withdrawn" = end match without this line plus line above
    '''
    lower_name = name.lower()

    # Collect all paragraphs/lines
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    current_name = ''
    reso_match = []

    # Find and store start of SLA section in Vote Sheet
    sla_index = 0
    for p in paragraphs:
        if "SLA Licensing & Outdoor Dining Committee".lower() in p.lower():
            sla_index = paragraphs.index(p)
            break
    if sla_index == 0:
        print("----- WARNING: NO SLA SECTION FOUND IN VOTE SHEET -----")

    for p in paragraphs:
        lower_p = p.lower()

        # If name found and end_name reached, return
        if end_name != None and current_name and end_name.lower() in lower_p:
            return '\n'.join(reso_match)
        # If name found and "withdrawn" line reached, return
        elif current_name and "withdrawn" in lower_p:
            reso_match = reso_match[:-1]    # remove previous line of text
            return '\n'.join(reso_match)
        elif current_name and "not heard at committee" in lower_p:
            return '\n'.join(reso_match)
        elif current_name and "vote to adjourn" in lower_p:
            return '\n'.join(reso_match)
        # If name found, save line of text
        elif lower_name in lower_p or current_name:
            current_name = lower_name
            reso_match.append(p)


        # # Ideal scenario where 
        # if end_name != None:
        #     # If name found and end_name reached, return
        #     if current_name and end_name.lower() in lower_p:
        #         return '\n'.join(reso_match)
        #     # If name found and "withdrawn" line reached, return
        #     elif current_name and "withdrawn" in lower_p:
        #         reso_match = reso_match[:-1]    # remove previous line of text
        #         return '\n'.join(reso_match)
        #     # If name found, save line of text
        #     elif lower_name in lower_p or current_name:
        #         current_name = lower_name
        #         reso_match.append(p)

        # # establish "base cases" at end of voted resos
        # else:
        #     # If name found and "withdrawn" line reached, return
        #     if current_name and "withdrawn" in lower_p:
        #         reso_match = reso_match[:-1]    # remove previous line of text
        #         return '\n'.join(reso_match)
        #     elif current_name and "not heard at committee" in lower_p:
        #         return '\n'.join(reso_match)
        #     elif current_name and "vote to adjourn" in lower_p:
        #         return '\n'.join(reso_match)
        #     # If name found, save line of text
        #     elif lower_name in lower_p or current_name:
        #         current_name = lower_name
        #         reso_match.append(p)
            


# ______________ FOLDERS & PATHS ___________________________________________ #
parent_folder = r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space'
sla_applications_folder = r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\SLA Application PDFs'

pdf_path = r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\Test-Application.pdf'
output_folder = r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\Text Outputs'

# Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)


# Images converted from PDFs (not required to save, but helpful for testing)
image_folder = r"C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\converted_images"

# Path to Vote Sheet / Minutes document with resolution texts
vote_sheet_path = r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\COPY_2025-05-Vote-Sheet.docx'
#vote_sheet_path = r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\test-vote-sheet.docx'

# __________________________________________________________________ #





'''
Perform OCR on each PDF, extract data from text, and store in dataframe
'''
# Setup dataframe
headers = ['applicant_name', 'street_address', 'representative_name(s)', 'representative_email(s)', 'resolution_text']
df = pd.DataFrame(columns=headers)

for filename in os.listdir(sla_applications_folder):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(sla_applications_folder, filename)
        base_name = os.path.splitext(filename)[0]   # filename without extension
        text_output_path = os.path.join(output_folder, f"{base_name}.txt")

        print(f"\nProcessing {filename}...")

        # Create list of images for each page of PDF
        pdf_images = convert_from_path(
            pdf_path,
            dpi=400,             # Image quality (dots per inch)
            fmt='png',          # Output format (e.g., 'png', 'jpeg')
        )

        all_text = ''
        for i, image in enumerate(pdf_images):
            # Create folder for converted images and specify image names
            image_path = os.path.join(image_folder, f'{filename}_page_{i + 1}.png')

            
            # Preprocess images to aid OCR
            preprocessed_image = preprocess_for_ocr(image)

            # Manually save images
            preprocessed_image.save(image_path)
            print(f"Saved: {image_path}")

            # Convert to text and combine text from all related images (pages of PDF)
            '''
            Notes on config options for pytesseract
            - these can affect how the characters are interpreted
            - https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
            '''
            all_text += pytesseract.image_to_string(preprocessed_image)
            # all_text += pytesseract.image_to_string(preprocessed_image, config='--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@.,:-() ')
            # all_text += pytesseract.image_to_string(preprocessed_image, lang='eng') # most used


        with open(text_output_path, 'w') as ocr_output:
            all_text.encode('utf-8').strip()
            ocr_output.write(all_text)

        print(f"Saved OCR text to {text_output_path}")

        # # Read from txt files
        # with open(r'C:\Users\MN03\Documents\Python Scripts\SLA Automation\Test Space\ocr_output.txt', 'r') as f:
        #     text = f.read()
        #     info = extract_info(text)
        #     print(info)

        # Extract info from text using regex
        info = extract_info(all_text, pdf_path)

        # Add row to dataframe
        df = pd.concat([df, pd.DataFrame([info])], ignore_index=True)   # wrap dictionary in list



# Loads Microsoft Word .docx file
doc = Document(vote_sheet_path)

# Get list of applicant names to search for resolutions
name_list = df['applicant_name'].tolist()

# Temporary dictionary to store resolution texts
reso_dict = {}

'''
For each applicant name, search .docx for specific resolution text.
Uses next_name to create end condition.
'''
for i in range(len(name_list)):
    search_name = name_list[i]

    # If there is another name, then use it
    if i + 1 < len(name_list):
        next_name = name_list[i + 1]
    else:
        next_name = None
    
    # Save resolution text to dataframe
    reso_dict[search_name] = find_reso(doc, search_name, next_name)







print("dict length = ", len(reso_dict))
print(name_list)
# for key, value in reso_dict.items():
#     print(f"{key}: {value}")
#     print("\n")





# Map temporary resolution dictionary to main dictionary using applicant name as key
df['resolution_text'] = df['applicant_name'].map(reso_dict)

# Create CSV from dataframe
df.to_csv(os.path.join(parent_folder, "SLA_app_info.csv"), mode='w', index=True, header=True)











# ______________ RUNTIME TRACKING - END _________________________ #
# Calculate program runtime
end_time = time.time()
elapsed_time = end_time - start_time

# Convert to hours, minutes, seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

# Build a readable string
time_parts = []
if hours:
    time_parts.append(f"{hours}h")
if minutes:
    time_parts.append(f"{minutes}m")
time_parts.append(f"{seconds:.2f}s")

print("_____________________________")
print("| Execution time:", ' '.join(time_parts) + " |")
# _________________________________________________________________ #