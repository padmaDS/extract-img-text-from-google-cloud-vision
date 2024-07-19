from azure.storage.blob import BlobServiceClient, ContentSettings
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import uuid
from google.cloud import vision
import requests
from io import BytesIO
from openai import OpenAI
from sklearn.cluster import KMeans
import logging
import csv
import io
load_dotenv()

app = Flask(__name__)

# Initialize the YOLO model
model = YOLO("owndatanewspaper.pt")

# Azure Blob Storage credentials
account_name = os.getenv("AZURE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_ACCOUNT_KEY")
container_name = os.getenv("AZURE_CONTAINER_NAME")
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# OpenAI client setup
api_key = os.getenv('OPENAI_API_KEY')

# Create BlobServiceClient
service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
container_client = service_client.get_container_client(container_name)

# Google Cloud Vision API setup
api_key_path = 'modular-glider-412712-b1446360cea1.json'
client1 = vision.ImageAnnotatorClient()
client = OpenAI(api_key=api_key)

# Track processed URLs to avoid duplication
processed_urls = set()

embed_model = 'text-embedding-ada-002'

def get_embedding(text, model=embed_model):
    text = text.replace("\n", " ")
    if len(text) > 2000:
        text = text[:2000]  # Truncate text to 2000 characters
    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    return embedding

def get_gpt4_embeddings(texts):
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings

def download_excel_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading Excel from URL: {str(e)}")

def extract_text_from_url(url):
    """Extracts text from an image URL using Google Cloud Vision API"""
    image = vision.Image()
    image.source.image_uri = url

    response = client1.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    extracted_text = texts[0].description if texts else ''
    return extracted_text

def upload_to_azure_blob(image_path, input_file_name, index, overwrite=True):
    blob_folder_name = input_file_name  # Folder named after the input file
    padded_index = str(index).zfill(4)  # Zero-pad the index to 4 digits
    unique_id = str(uuid.uuid4())  # Generate a unique identifier
    blob_name = f"{blob_folder_name}/{input_file_name}_cropped_{padded_index}_{unique_id}.jpg"
    print(f'Uploading {blob_name} to Azure Blob Storage...')
    with open(image_path, 'rb') as data:
        try:
            container_client.upload_blob(name=blob_name, data=data, content_settings=ContentSettings(content_type='image/jpeg'), overwrite=overwrite)
            print(f'Successfully uploaded {blob_name}')
            return blob_name
        except Exception as e:
            print(f'Error uploading {blob_name}: {str(e)}')
            return None

def upload_excel_file(file_path, container_name, blob_name, overwrite=True):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    
    if not container_client.exists():
        container_client.create_container()

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    with open(file_path, 'rb') as data:
        blob_client.upload_blob(data, blob_type="BlockBlob", content_settings=ContentSettings(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'), overwrite=overwrite)
    
    blob_url = blob_client.url
    return blob_url

def upload_text_file(file_path, container_name, blob_name, overwrite=True):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(file_path, 'rb') as data:
        blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=overwrite)
    blob_url = blob_client.url
    return blob_url

@app.route('/newspaper-image-text', methods=['POST'])
def detect():
    if 'urls' not in request.json:
        return jsonify({"error": "No URLs provided in JSON payload"}), 400

    urls = request.json.get('urls')

    if not urls:
        return jsonify({"error": "Invalid URLs provided"}), 400

    all_blob_urls = []
    all_text_outputs = []  # List to store the text outputs for each URL
    main_dir = 'yolo_outputs'
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    run_dir = os.path.join(main_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    try:
        # Clear processed_urls set for each request
        processed_urls.clear()

        for url in urls:
            if url in processed_urls:
                print(f"URL {url} already processed, skipping...")
                continue

            processed_urls.add(url)  # Mark URL as processed

            input_file_name = os.path.splitext(os.path.basename(url))[0]  # Get the file name without extension

            # Download image from URL
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img)  # Convert image to numpy array for processing

            # Run YOLO inference
            results = model.predict(img_np, imgsz=320, conf=0.10, save_crop=True)

            cropped_images = []

            # Draw bounding boxes and save cropped images
            for i, result in enumerate(results):
                for j, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list and then to int
                    confidence = box.conf.item()  # Extract confidence score
                    label = f'{box.cls.item()}: {confidence:.2f}'  # Extract class and confidence
                    color = (0, 255, 0)  # Green color for bounding boxes
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Crop and save the region
                    cropped_img = img_np[y1:y2, x1:x2]
                    cropped_images.append(cropped_img)

            local_blob_urls = []

            # Upload cropped images to Azure Blob Storage and collect blob URLs
            for idx, cropped_img in enumerate(cropped_images):
                temp_cropped_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(temp_cropped_file.name, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                blob_name = upload_to_azure_blob(temp_cropped_file.name, input_file_name, idx)
                if blob_name:
                    blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"
                    local_blob_urls.append(blob_url)

            all_blob_urls.extend(local_blob_urls)

            # Extract text from each URL
            for local_blob_url in local_blob_urls:
                try:
                    extracted_text = extract_text_from_url(local_blob_url)
                    all_text_outputs.append({"url": local_blob_url, "extracted_text": extracted_text})
                except Exception as e:
                    all_text_outputs.append({"url": local_blob_url, "extracted_text": f"Error extracting text: {str(e)}"})

            # Save blob URLs to a text file in the subfolder
            urls_file_path = os.path.join(run_dir, f'{input_file_name}_blob_urls.txt')
            with open(urls_file_path, 'w', encoding='utf-8') as urls_file:
                for url in local_blob_urls:
                    urls_file.write(url + '\n')

            # Create DataFrame to store data
            df = pd.DataFrame({
                'Input Image': [input_file_name] * len(local_blob_urls),
                'Blob Links': local_blob_urls
            })

            # Save DataFrame to Excel
            excel_file_path = os.path.join(run_dir, f'{input_file_name}_blob_urls.xlsx')
            df.to_excel(excel_file_path, index=False, engine='openpyxl')

        # Create the final JSON response
        final_response = {
            "num_cropped_images": len(all_blob_urls),
            "blob_urls": all_blob_urls,
            "text_outputs": all_text_outputs,
            "folder_path": run_dir
        }

        ###Save the final JSON response fields to a text file
         # Define the file path where you want to save the text file
        newspaper_content_file_path = os.path.join(final_response["folder_path"], 'newspaper_content.txt')

        # Debug: Print final_response to ensure it's correct
        print("Final Response:", final_response)

        # Open the file in write mode with UTF-8 encoding
        with open(newspaper_content_file_path, 'w', encoding='utf-8') as txt_file:
            # Write only the 'text_outputs' part
            txt_file.write("text_outputs:\n")
            for item in final_response["text_outputs"]:
                # Debug: Print each item being written
                print(f"Writing item: url: {item['url']}, extracted text: {item['extracted_text']}")
                txt_file.write(f"url: {item['url']}\nextracted text: {item['extracted_text']}\n\n")

        # Upload the text file to Azure Blob Storage
        blob_name = 'telugu_textfile.txt'  # Define the name of the text file in the blob
        text_file_blob_url = upload_text_file(newspaper_content_file_path, container_name, blob_name)
        final_response["text_file_blob_url"] = text_file_blob_url

        # Create DataFrame for text outputs
        text_df = pd.DataFrame(final_response["text_outputs"])

        # Save DataFrame to CSV
        excel_file_path = os.path.join(run_dir, 'text_outputs.xlsx')
        text_df.to_excel(excel_file_path, index=False)

        # Upload the CSV file to Azure Blob Storage
        excel_blob_name = 'telugu_text_outputs.xlsx'
        excel_blob_url = upload_excel_file(excel_file_path, container_name, excel_blob_name)
        final_response["excel_file_blob_url"] = excel_blob_url

        # Upload the text file to Azure Blob Storage
        blob_name = 'telugu_textfile.txt'  # Define the name of the text file in the blob
        text_file_blob_url = upload_text_file(newspaper_content_file_path, container_name, blob_name)
        final_response["text_file_blob_url"] = text_file_blob_url

        return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cluster-text', methods=['POST'])
def cluster_text():
    data = request.json

    if not data or 'text_blob_url' not in data:
        return jsonify({"error": "Missing text_blob_url in JSON payload"}), 400

    text_blob_url = data['text_blob_url']

    def download_text_from_url(url):
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def get_completion(prompt, model="gpt-4o"):
        messages = [{"role": "user", "content": prompt}]
        # Ensure 'client' is properly defined for GPT-4o API interaction
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7  # Adjust temperature as needed
        )
        return completion.choices[0].message.content.splitlines()  # Split by lines for clusters

    try:
        extracted_texts = download_text_from_url(text_blob_url)

        details1 = f"""
        You will be given the extracted text from the Telugu News Paper in triple quotes.
        Your job is to form the cluster based on the given information in Telugu Language along with respective blob urls and Telugu text.
        '''{extracted_texts}'''
        """

        clusters = get_completion(details1)

        # Directory for saving files locally
        save_dir = os.path.join(os.getcwd(), 'cluster_results')  # Using current working directory
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        # Save clusters to a CSV file
        csv_filename = f"clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(save_dir, csv_filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Cluster'])
            csv_writer.writerows([[cluster] for cluster in clusters])

        # Save clusters to a TXT file
        txt_filename = f"clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        txt_path = os.path.join(save_dir, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as txtfile:
            txtfile.write('\n\n'.join(clusters))

        # Upload CSV file to Azure Blob Storage
        blob_client_csv = container_client.get_blob_client(csv_filename)
        with open(csv_path, "rb") as data:
            blob_client_csv.upload_blob(data, overwrite=True)

        # Upload TXT file to Azure Blob Storage
        blob_client_txt = container_client.get_blob_client(txt_filename)
        with open(txt_path, "rb") as data:
            blob_client_txt.upload_blob(data, overwrite=True)

        # Get URLs of uploaded files
        csv_blob_url = blob_client_csv.url
        txt_blob_url = blob_client_txt.url

        return jsonify({
            "clusters": clusters,
            "txt_blob_url": txt_blob_url
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/cluster-text-kmeans', methods=['POST'])
def cluster_text_from_blob_url():
    # Get JSON data from POST request
    request_data = request.get_json()

    # Validate and extract Excel blob URL
    excel_blob_url = request_data.get('excel_blob_url')
    if not excel_blob_url:
        return jsonify({'error': 'Excel blob URL missing in request'}), 400

    # Validate and extract number of clusters
    num_clusters = request_data.get('num_of_clusters')
    if not num_clusters or not isinstance(num_clusters, int):
        return jsonify({'error': 'Number of clusters missing or invalid'}), 400

    # Download Excel content from Azure Blob Storage using the blob URL
    try:
        excel_content = download_excel_from_url(excel_blob_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Read Excel into pandas DataFrame
    try:
        df = pd.read_excel(io.BytesIO(excel_content))  # Use io.BytesIO for compatibility
    except Exception as e:
        return jsonify({'error': f'Error reading Excel: {str(e)}'}), 500

    # Extract the text column (adjust this based on your Excel structure)
    articles = df['extracted_text'].tolist()

    # Get embeddings for the articles (replace this with your actual embedding function)
    embeddings = get_gpt4_embeddings(articles)

    # Add embeddings to the DataFrame (adjust this based on your actual embedding function)
    df['embedding'] = embeddings

    # Initialize K-Means with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit K-Means on the embeddings
    kmeans.fit(embeddings)

    # Get cluster labels for each article
    labels = kmeans.labels_

    # Add cluster labels to the DataFrame
    df['cluster'] = labels

    # Save the clustered DataFrame to a new Excel file (optional)
    output_file_path = "clustered_news_articles.xlsx"
    df.to_excel(output_file_path, index=False)

    response_text = df.to_string(index=False)

    # Convert DataFrame to JSON for response (optional)
    # result_json = df.to_json(orient="records")
    # Upload the file to Azure Blob Storage

    blob_client = container_client.get_blob_client(output_file_path)
    with open(output_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    # Get the Blob URL
    cluster_blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{output_file_path}"

    # Return plain text response with the formatted DataFrame
 #   return jsonify({response_text, 'blob_url': blob_url})
    # return jsonify({'response_text': response_text, 'blob_url': blob_url})

    return jsonify({"response" : response_text, 'cluster_blob_url': cluster_blob_url})


    # return jsonify({'result': result_json})

if __name__ == '__main__':
   
    app.run(host='localhost', port=8000, debug=True)