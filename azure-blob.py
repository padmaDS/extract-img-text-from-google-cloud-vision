import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from dotenv import load_dotenv
load_dotenv()

account_name = os.getenv("AZURE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_ACCOUNT_KEY")
container_name = os.getenv("AZURE_CONTAINER_NAME")
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
print(connect_str)

def upload_text_file(file_path, container_name, blob_name):

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create the container if it does not exist
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()

    # Create a blob client using the file name
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Upload the file
    with open(file_path, 'rb') as data:
        blob_client.upload_blob(data, blob_type="BlockBlob")

    # Get the URL of the uploaded blob
    blob_url = blob_client.url
    return blob_url

if __name__ == "__main__":
    file_path = r'yolo_outputs\2024-07-15_17-56-18\newspaper_content.txt'  # Replace with your local file path
    container_name = container_name  # Replace with your container name
    blob_name = 'telugu_textfile.txt'  # Replace with the desired blob name

    blob_url = upload_text_file(file_path, container_name, blob_name)
    print(f"File uploaded to: {blob_url}")
