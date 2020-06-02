from storageservice import StorageClient
import os

def test_savefile():
    os.environ["AzureWebJobsStorage"]="DefaultEndpointsProtocol=https;AccountName=cekoiainvoiceextractor;AccountKey=oqOQCs2PGlxt6U/AkW6TTFr44nMi5ZZmaH6uNLACxUt8Tl242f7K44RaiGLdUnvIyQGkxO8o0ACf+QbS1NDPSg==;EndpointSuffix=core.windows.net"
    client=StorageClient()
    client.savefile('clienttest','test.dat',"test;base64,test")

def test_findallcustomers():
    os.environ["AzureWebJobsStorage"]="DefaultEndpointsProtocol=https;AccountName=cekoiainvoiceextractor;AccountKey=oqOQCs2PGlxt6U/AkW6TTFr44nMi5ZZmaH6uNLACxUt8Tl242f7K44RaiGLdUnvIyQGkxO8o0ACf+QbS1NDPSg==;EndpointSuffix=core.windows.net"
    client=StorageClient()
    customers=client.findallcustomers()
    assert len(customers)>0
