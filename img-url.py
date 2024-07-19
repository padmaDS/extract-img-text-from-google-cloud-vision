from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
import pandas as pd
from dotenv import load_dotenv
import uuid  # For generating unique identifiers
from google.cloud import vision
import io
 
load_dotenv()
 
app = Flask(__name__)
 
# Initialize the YOLO model
model = YOLO("owndatanewspaper.pt")
 
# Azure Blob Storage credentials
account_name = os.getenv("AZURE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_ACCOUNT_KEY")
container_name = os.getenv("AZURE_CONTAINER_NAME")
 
# Create BlobServiceClient
service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
container_client = service_client.get_container_client(container_name)
 
# Google Cloud Vision API setup
api_key_path = 'modular-glider-412712-b1446360cea1.json'
client = vision.ImageAnnotatorClient()
 
def extract_text_from_blob_url(blob_url):
    """Extracts text from an image blob URL using Google Cloud Vision API"""
    image = vision.Image()

    image.source.image_uri = blob_url

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    extracted_text = texts[0].description if texts else ''
    return extracted_text

def upload_to_azure_blob(image_path, input_file_name, index):
    blob_folder_name = input_file_name  # Folder named after the input file
    padded_index = str(index).zfill(4)  # Zero-pad the index to 4 digits
    unique_id = str(uuid.uuid4())  # Generate a unique identifier
    blob_name = f"{blob_folder_name}/{input_file_name}_cropped_{padded_index}_{unique_id}.jpg"
    print(f'Uploading {blob_name} to Azure Blob Storage...')
    with open(image_path, 'rb') as data:
        try:
            container_client.upload_blob(name=blob_name, data=data, content_settings=ContentSettings(content_type='image/jpeg'))
            print(f'Successfully uploaded {blob_name}')
            return blob_name
        except Exception as e:
            print(f'Error uploading {blob_name}: {str(e)}')
            return None

@app.route('/newspaper-image-text', methods=['POST'])
def detect():
    if 'blob_urls' not in request.json:
        return jsonify({"error": "No blob URLs provided in JSON payload"}), 400

    blob_urls = request.json.get('blob_urls')

    if not blob_urls:
        return jsonify({"error": "Invalid blob URLs provided"}), 400

    all_blob_urls = []
    all_text_outputs = []  # List to store the text outputs for each URL
    main_dir = 'yolo_outputs'
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    run_dir = os.path.join(main_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    try:
        for blob_url in blob_urls:
            input_file_name = os.path.splitext(os.path.basename(blob_url))[0]  # Get the file name without extension

            # Download image from blob URL (optional depending on processing flow)
            # Example: You might want to download the image temporarily to process locally

            # Run YOLO inference (example code using local image processing)
            # For blob URL processing, you might need to adapt how YOLO is applied

            # Extract text from the blob URL using Google Cloud Vision API
            try:
                extracted_text = extract_text_from_blob_url(blob_url)
                all_text_outputs.append({"url": blob_url, "extracted_text": extracted_text})
            except Exception as e:
                all_text_outputs.append({"url": blob_url, "extracted_text": f"Error extracting text: {str(e)}"})

            all_blob_urls.append(blob_url)

        # Create the final JSON response
        final_response = {
            "num_cropped_images": len(all_blob_urls),
            "blob_urls": all_blob_urls,
            "text_outputs": all_text_outputs,
            "folder_path": run_dir
        }

        # Save the final JSON response fields to a text file
        newspaper_content_file_path = os.path.join(run_dir, 'newspaper_content.txt')
        with open(newspaper_content_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write("blob_urls:\n")
            for url in final_response["blob_urls"]:
                txt_file.write(f"{url}\n")
            txt_file.write("\ntext_outputs:\n")
            for item in final_response["text_outputs"]:
                txt_file.write(f"url: {item['url']}\nextracted text: {item['extracted_text']}\n\n")

        # Return JSON response with processed image count, blob URLs, and the folder path
        return jsonify(final_response)

    except Exception as e:
        error_message = f"Error processing blob URLs: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)

   
