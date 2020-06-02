from dbservice import DbClient
import os

def test_findallinvoices():
    os.environ['ACCOUNT_URI']="https://cekoiadev-cosmosdb.documents.azure.com:443/"
    os.environ["ACCOUNT_KEY"]="XmbjB33akVB5uVU4R293MdibqOPLThei1XO9SymOMwNro9GchYSncAlDOsOQHsRm9GFc4HVrG2fiZGwUhxzbkA=="
    client=DbClient()
    invoices=client.findallinvoices()
    assert len(invoices)>0

def test_findallcontrols():
    os.environ['ACCOUNT_URI']="https://cekoiadev-cosmosdb.documents.azure.com:443/"
    os.environ["ACCOUNT_KEY"]="XmbjB33akVB5uVU4R293MdibqOPLThei1XO9SymOMwNro9GchYSncAlDOsOQHsRm9GFc4HVrG2fiZGwUhxzbkA=="
    client=DbClient()
    controls=client.findallinvoices()
    assert len(controls)>0