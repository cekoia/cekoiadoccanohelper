import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64
import logging
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.errors as errors
import azure.cosmos.http_constants as http_constants
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient

UPLOAD_DIRECTORY='/tmp'

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

connect_str= os.environ["AzureWebJobsStorage"]
url = os.environ['ACCOUNT_URI']
key = os.environ['ACCOUNT_KEY']
client = cosmos_client.CosmosClient(url_connection=url, auth={"masterKey":key})
database_id='cekoia'
container_id='invoices'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server

app.layout = html.Div(children=[
    html.H4(children='Invoices loader'),
    dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop your pdf invoice file to convert."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.Div(id='placeholder'),
         html.H4(children='Invoices'),
        html.Div(id='live-update-table'),
        html.H4(children='Invoices details'),
        html.Div(id='live-update-details'),
        dcc.Interval(
                id='interval-component',
                interval=1*1000, # in milliseconds
                n_intervals=0
            )
])

def uploadtoazure(localpath, remotedir, connect_str):
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
  blob_service_client = BlobServiceClient.from_connection_string(connect_str)
  blob_client = blob_service_client.get_blob_client(container=containername, blob=remotepath)
  with open(localpath, "rb") as data:
      blob_client.upload_blob(data,overwrite=True)

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    localpath=os.path.join(UPLOAD_DIRECTORY, name)
    print(localpath)
    with open(localpath, "wb") as fp:
        fp.write(base64.decodebytes(data))
    uploadtoazure(localpath, 'customers/clienttest/in',connect_str)
    os.remove(localpath)
    print('file sent')
    
def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(filename)
    return html.A(filename, href=location)

@app.callback(
    Output("placeholder", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li("ok")]

@app.callback([Output('live-update-table', 'children'),Output('live-update-details', 'children'),],
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    df=pd.DataFrame(client.QueryItems("dbs/" + database_id + "/colls/" + container_id,'SELECT * FROM c'))
    
    details=[]
    for d in df.details.tolist():
        details.extend(d)
    details=pd.DataFrame(details)
    #print(details)
    df=df.drop(['details','_rid','_self','_etag','_attachments','_ts','id'],1)
    max_rows=20
    invoices= html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(len(df))
        ])
    ])
    det=html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in details.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(details.iloc[i][col]) for col in details.columns
                ]) for i in range(len(details))
            ])
    ])
    return invoices,det

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port='80')