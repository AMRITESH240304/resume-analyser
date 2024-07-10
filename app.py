import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
import cv2
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

def extract_text(image):
    return pytesseract.image_to_string(image)

def pdf_to_image():
    pdf_path = 'Amrtitesh-Kumar.pdf'
    pages = convert_from_path(pdf_path, 300, first_page=1, last_page=1)
    if pages:
        first_page_image = pages[0]
        output_folder = 'output_image'
        os.makedirs(output_folder, exist_ok=True)
        image_path = os.path.join(output_folder, 'first_page.jpg')
        first_page_image.save(image_path, 'JPEG')
        print(f"First page of the PDF has been converted to an image and saved as '{image_path}'.")
    else:
        print("The PDF is empty or the specified page range is invalid.")

def content_extract(image, word):
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if data['text'][i].lower() == word.lower():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            print(f"Found word '{word}' at position ({x}, {y}, {w}, {h})")
    cv2.imshow('Detected Word', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_section(text, start_word, end_word):
    start_index = text.lower().find(start_word.lower())
    end_index = text.lower().find(end_word.lower(), start_index)
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index + len(end_word)] 

def full_ocr(image):
    text_folder = 'text_file'
    os.makedirs(text_folder,exist_ok=True)
    fileName = 'full_text.txt'
    text_path = os.path.join(text_folder,fileName)
    
    with open(text_path,'w') as file:
        file.write(text)
        
    print(f'file saved successfully')

def save_to_txt(text):
    text_folder = 'text_file'
    os.makedirs(text_folder,exist_ok=True)
    fileName = 'text.txt'
    text_path = os.path.join(text_folder,fileName)
    
    with open(text_path,'w') as file:
        file.write(text)
        
    print(f'file saved successfully')
    

pdf_to_image()  

Pages = ['output_image/first_page.jpg']  
start_word = 'Skills'
end_word = 'Projects'

full_text = ""

for page_path in Pages:
    binary_image = process_image(page_path)
    text = extract_text(binary_image)
    full_text += text + "\n"
    
full_ocr(full_text)
section_text = find_section(full_text, start_word, end_word)

if section_text:
    # print("Extracted Section:\n", section_text)
    save_to_txt(section_text)
    print(os.getenv('MISTRAL_AI_API_KEY'))
else:
    print(f"The section starting with '{start_word}' and ending with '{end_word}' was not found.")
    
