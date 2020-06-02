from azure.storage.blob import BlobServiceClient,ContainerClient
import logging
import os
import base64

class StorageClient:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])
        self.container = ContainerClient.from_connection_string(os.environ["AzureWebJobsStorage"], container_name="customers")
        self.UPLOAD_DIRECTORY='/tmp'
        if not os.path.exists(self.UPLOAD_DIRECTORY):
            os.makedirs(self.UPLOAD_DIRECTORY)

    def uploadtoazure(self,localpath, remotedir):
        """
        upload un fichier vers azure
        @localpath: chemin du fichier en local
        @remotedir: répertoire de copie distant
        @connect_str: chaîne de connexion au bucket
        """
        localfilename=localpath.split('/')[-1]
        splits=remotedir.split("/")
        containername=splits[0]
        remotepath='/'.join(splits[1:])+'/'+localfilename
        logging.info(f'uploading {localpath} to {remotepath}')
        
        blob_client = self.blob_service_client.get_blob_client(container=containername, blob=remotepath)
        with open(localpath, "rb") as data:
            blob_client.upload_blob(data,overwrite=True)

    def savefile(self,customer,name, content):
        """Decode and store a file uploaded with Plotly Dash."""
        data = content.encode("utf8").split(b";base64,")[1]
        localpath=os.path.join(self.UPLOAD_DIRECTORY, name)
        print(localpath)
        with open(localpath, "wb") as fp:
            fp.write(base64.decodebytes(data))
        self.uploadtoazure(localpath, f'customers/{customer}/in')
        os.remove(localpath)
        print('file sent')

    def findallcustomers(self):
        blob_list = self.container.list_blobs()
        customers=set()
        for blob in blob_list:
           customers.add(blob.name.split('/')[0])
        return customers