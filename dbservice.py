import azure.cosmos.cosmos_client as cosmos_client
import pandas as pd
import os 

class DbClient:
    def __init__(self):
        url = os.environ['ACCOUNT_URI']
        key = os.environ['ACCOUNT_KEY']
        client = cosmos_client.CosmosClient(url, key)
        db = client.get_database_client('cekoia')
        self.invoices = db.get_container_client('invoices')
        self.controls = db.get_container_client('controls')

    def findallinvoices(self):
        items=list(self.invoices.read_all_items(max_item_count=100))
        results= pd.DataFrame(items)
        return results.drop(['_rid','_self','_etag','_attachments','_ts','id'],1)

    def findallcontrols(self):
        items=list(self.controls.read_all_items(max_item_count=100))
        results= pd.DataFrame(items)
        return results[["key","anomaly","invoiceid","customer"]]

    