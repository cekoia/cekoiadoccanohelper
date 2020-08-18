import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import logging
import pandas as pd
import os
from service import *

def set_projects_options():
    global resource
    return [{'label': name, 'value': name} for name in findallprojects(resource)] 

connect_str=os.environ['AzureWebJobsStorage']
resource=connect_str.split(";")[1].split("=")[1].replace('cekoia','').replace('storage','')

localannotationpath=""
df=pd.DataFrame()
app = dash.Dash(__name__)
#application = app.server

app.layout = html.Div(children=[
   html.H4(children=f'Environnement: {resource}'),
    html.H4(children='Sélectionner un projet doccano'),
    dcc.RadioItems(
        id='project',options=set_projects_options()
    ),
    html.Button('Importer les annotations',id='import'),
    html.Button('Exporter les annotations', id='export'),
    html.Button('Générer le modèle', id='model'),
    html.Div(id='importstatus'),
    html.Div(id='exportstatus'),
    html.Div(id='modelstatus'),
    html.Div(id='tabs',children=[dcc.Tabs([
        dcc.Tab(label='Contrôles de cohérence', children=[
            html.P("permet de repérer pour chaque label des valeurs bizarres, afin de les corriger manuellement dans doccano en se basant sur le docid"),
            html.Div(id='outliers')
        ]),
        dcc.Tab(label='Statistiques de remplissage', children=[
            html.P("..."),
            html.Div(id='report')
        ]),
        dcc.Tab(label='Enrichissement d\'annotations par recopie', children=[
            html.P("Le traitement repère les annotations toujours identiques, telles que des adresses ou des numéros de tvas, et les annote automatiquement dans les documents nécessaires"),
            html.Button(
            'Lancer le traitement',
            id='isoautocomplete',
        ),
        html.Div(id='isoautocompletestatus'),
        ]),
        dcc.Tab(label='Enrichissement d\'annotations par modélisation', children=[
            html.P("Le traitement modélise les annotations que vous avez seulement faites dans certains documents, telles que des codes tvas, et les annote automatiquement dans les documents nécessaires"),
            html.Button(
            'Lancer le traitement',
            id='autocomplete',
        ),
        html.Div(id='autocompletestatus'),
        ]),
    ])])
])

def generatereports(df):
    outliers=findlocaloutliers(df)
    if len(outliers)>0:
        outliers=dash_table.DataTable(
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        columns=[{"name": i, "id": i} for i in outliers.columns],
        data=outliers.to_dict('records'),)
    else:
        outliers=""
    reporting=report(df)
    if len(reporting)>0:
        reporting['docids']=reporting.docids.astype(str)
        reporting=dash_table.DataTable(
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        columns=[{"name": i, "id": i} for i in reporting.columns],
        data=reporting.to_dict('records'),)
    else:
        reporting=""
    return outliers,reporting
    




@app.callback(
    Output('project', 'value'),
    [Input('project', 'options')])
def set_cities_value(available_options):
    return available_options[0]['value']

@app.callback(
    [Output(component_id='importstatus', component_property='children'),Output(component_id='outliers', component_property='children'),Output(component_id='report', component_property='children'),Output('tabs','style')],
    [Input(component_id='project', component_property='value'),Input('import','n_clicks')]
)
def importannotations(customer,n_clicks):
    if n_clicks!=None:
        global df
        global localannotationpath
        global resource
        df,localannotationpath=importdoccanoannotations(resource, customer)
        outliers,report=generatereports(df)

        return f'{len(df)} annotations importées depuis le projet {customer}',outliers,report,{'visibility':'visible'}
    else:
        return '','','',{'visibility':'hidden'}
@app.callback(
    Output(component_id='exportstatus', component_property='children'),
    [Input(component_id='project', component_property='value'),Input('export','n_clicks')]
)
def exportannotations(customer,n_clicks):
    if n_clicks!=None:
        global df
        global resource
        exportdoccanoannotations(resource,customer,df)
        return f'{len(df)} annotations exportées depuis le projet {customer}'
    else:
        return ''

@app.callback(
    Output(component_id='isoautocompletestatus', component_property='children'),
    [Input(component_id='isoautocomplete', component_property='n_clicks')]
)
def launchisoautocomplete(n_clicks):
    global df
    if n_clicks!=None:
        before = df.copy()
        df=annotatefixedlabels(before)
        summary=""
        if len(df)>len(before):
            for _,row in df.tail(n=len(df)-len(before)).iterrows():
                summary+=row+"<br/>"
        else:
            summary="aucune annotation ajoutée"
        return summary
    else:
        return ""

@app.callback(
    Output(component_id='autocompletestatus', component_property='children'),
    [Input(component_id='autocomplete', component_property='n_clicks')]
)
def launchautocomplete(n_clicks):
    global df
    global localannotationpath
    if n_clicks!=None:
        before = df.copy()
        df=autocompletedocs(df,localannotationpath)
        summary=""
        if len(df)>len(before):
            for _,row in df.tail(n=len(df)-len(before)).iterrows():
                summary+=row+"<br/>"
        else:
            summary="aucune annotation ajoutée"
        return summary
    else:
        return ""

@app.callback(
    Output(component_id='modelstatus', component_property='children'),
    [Input(component_id='project', component_property='value'),Input('model','n_clicks')]
)
def generatemodel(customer,n_clicks):
    if n_clicks!=None:
        global connect_str
        train('annotation.jsonl',connect_str,customer,localdir='.')
        return "modèle créé et copié sur Azure"
    else:
        return ""

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0",port=5000)