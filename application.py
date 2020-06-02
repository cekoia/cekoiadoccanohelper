import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import logging
import pandas as pd
from storageservice import StorageClient
from dbservice import DbClient

def getstyle(col, key):
    if key==col:
        return {'background-color':'red'}
    else:
        return {}

dbclient=DbClient()
storageclient=StorageClient()




app = dash.Dash(__name__)
application = app.server

app.layout = html.Div(children=[
   html.H4(children='Select a customer'),
   dcc.Dropdown(
        id='customer',
        options=[{'label': customer, 'value': customer} for customer in storageclient.findallcustomers()] , value='clientserieux'      
    ),
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


@app.callback(
    Output("placeholder", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            storageservice.savefile('clientserieux',name, data)

    return [html.Li("ok")]

@app.callback([Output('live-update-table', 'children'),Output('live-update-details', 'children'),],
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    df=dbclient.findallinvoices()
    controls=dbclient.findallcontrols()
    df=df.merge(controls, how='left')
    details=[]
    for d in df.details.tolist():
        details.extend(d)
    details=pd.DataFrame(details)
    invoices= html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns if col not in ['customer','details']])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col], style=getstyle(col, df.iloc[i]['key'])) for col in df.columns if col not in ['customer','details']
            ]) for i in range(len(df))
        ])
    ])
    det=html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in details.columns] )
            ),
            html.Tbody([
                html.Tr([
                    html.Td(details.iloc[i][col]) for col in details.columns
                ]) for i in range(len(details))
            ])
    ])
    return invoices,det

if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0', port='80')