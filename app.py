import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import dash_table
import plotly.express as px

import logging
import pandas as pd
import os
from service import *


connect_str=os.environ['AzureWebJobsStorage']
resource=connect_str.split(";")[1].split("=")[1].replace('cekoia','').replace('storage','')
projects_options=[{'label': name, 'value': name} for name in findallprojects(resource)] 
localannotationpath=""
df=pd.DataFrame()

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(children=[
    html.H4(children='Version modele 1 '),
   html.H4(children=f'Environnement: {resource}'),
    html.H4(children='Sélectionner un projet doccano'),
    dcc.RadioItems(
        id='project',options=projects_options
    ),
    html.Button('Importer les annotations',id='import'),
    html.Button('Enregistrer', id='export'),
    html.Button('Générer le modèle', id='model'),
    dcc.Loading(id="loading-1",
            type="default",
        children=[html.Div(id='importstatus'),html.Div(id='modelstatus'),html.Div(id='isoautocompletestatus'),html.Div(id='autocompletestatus')]),
    dcc.Graph(id="graph"),
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
        
        ]),
        dcc.Tab(label='Enrichissement d\'annotations par modélisation', children=[
            html.P("Le traitement modélise les annotations que vous avez seulement faites dans certains documents, telles que des codes tvas, et les annote automatiquement dans les documents nécessaires"),
            html.Button(
            'Lancer le traitement',
            id='autocomplete',
        )
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

def generategraphfigure(df):
    stats=df.groupby(['docid','label']).start.count().reset_index().rename(columns={'start':'annotations'})
    fig = px.bar(stats, x="docid", y="annotations", color="label", title="Annotations par documents")
    return fig

@app.callback(
    [Output('importstatus','children'),Output('outliers', 'children'),Output('report', 'children'),Output('tabs','style'),Output('graph','figure'),Output('graph','style')],
    [Input('import','n_clicks'),Input('export','n_clicks')],
    [State('project', 'value')]
)
def importannotations(importclic,exportclic,customer):
    global df
    global localannotationpath
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id=='import':
        df,localannotationpath=importdoccanoannotations(resource, customer)
        outliers,report=generatereports(df)
        fig=generategraphfigure(df)
        return f'{len(df)} annotations importées depuis le projet {customer}',outliers,report,{'visibility':'visible'},fig,{'visibility':'visible'}
    elif button_id=='export':
        exportdoccanoannotations(resource,customer,df)
        outliers,report=generatereports(df)
        fig=generategraphfigure(df)
        return f'{len(df)} annotations exportées depuis le projet {customer}',outliers,report,{'visibility':'visible'},fig,{'visibility':'visible'}
    else:
        return '','','',{'visibility':'hidden'},{},{'visibility':'hidden'}

@app.callback(
    Output('isoautocompletestatus', 'children'),
    [Input('isoautocomplete', 'n_clicks')]
)
def launchisoautocomplete(n_clicks):
    global df
    if n_clicks!=None:
        before = df.copy()
        df=annotatefixedlabels(before)
        summary=""
        if len(df)>len(before):
            n=len(df)-len(before)
            summary=f'ajout automatique de {n} annotations'
        else:
            summary="aucune annotation ajoutée"
        return summary
    else:
        return ""

@app.callback(
    Output('autocompletestatus', 'children'),
    [Input('autocomplete', 'n_clicks')]
)
def launchautocomplete(n_clicks):
    global df
    if n_clicks!=None:
        before = df.copy()
        df=autocompletedocs(df,localannotationpath)
        
        summary=""
        if len(df)>len(before):
            n=len(df)-len(before)
            summary=f'ajout automatique de {n} annotations'
        else:
            summary="aucune annotation ajoutée"
        return summary
    else:
        return ""

@app.callback(
    Output('modelstatus', 'children'),
    [Input('model','n_clicks')],
    [State('project', 'value')]
)
def generatemodel(n_clicks,customer):
    if n_clicks!=None:
        global connect_str
        train('annotation.jsonl',connect_str,customer,localdir='/tmp')
        return "modèle créé et copié sur Azure"
    else:
        return ""

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080) 