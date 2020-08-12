import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import logging
import pandas as pd
import os
from service import *

resources=['dev','qual']
localannotationpath=""
df=pd.DataFrame()
app = dash.Dash(__name__)
application = app.server

app.layout = html.Div(children=[
   html.H4(children='Sélectionner un environnement'),
   dcc.RadioItems(
        id='resource',
        options=[{'label': resource, 'value': resource} for resource in resources] , value='qual'      
    ),
    html.H4(children='Sélectionner un projet doccano'),
    dcc.RadioItems(
        id='project'
    ),
    html.Div(id='status'),
    html.Div([dcc.Tabs([
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
        dcc.Tab(label='Contrôles de cohérence', children=[
            html.P("permet de repérer pour chaque label des valeurs bizarres, afin de les corriger manuellement dans doccano en se basant sur le docid"),
            html.Div(id='outliers')
        ]),
        dcc.Tab(label='Statistiques de remplissage', children=[
            html.P("..."),
            html.Div(id='report')
        ]),
    ])])
])

def generatereports(df):
    outliers=findlocaloutliers(df)
    reporting=report(df)
    return dash_table.DataTable(
    style_cell={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    columns=[{"name": i, "id": i} for i in outliers.columns],
    data=outliers.to_dict('records'),),dash_table.DataTable(
    style_cell={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    columns=[{"name": i, "id": i} for i in reporting.columns],
    data=reporting.to_dict('records'),)

@app.callback(
    Output(component_id='project', component_property='options'),
    [Input(component_id='resource', component_property='value')]
)
def set_projects_options(selected_resource):
    return [{'label': name, 'value': name} for name in findallprojects(selected_resource)] 

@app.callback(
    Output('project', 'value'),
    [Input('project', 'options')])
def set_cities_value(available_options):
    return available_options[0]['value']

@app.callback(
    [Output(component_id='status', component_property='children'),Output(component_id='outliers', component_property='children'),Output(component_id='report', component_property='children')],
    [Input(component_id='resource', component_property='value'),Input(component_id='project', component_property='value')]
)
def importannotations(resource,customer):
    logging.info(f'resource = {resource} customer = {customer}')
    global df
    global localannotationpath
    df,localannotationpath=importdoccanoannotations(resource, customer)
    outliers,report=generatereports(df)
    return f'imported {len(df)} annotations successfully!',outliers,report

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

if __name__ == '__main__':
    application.run(debug=True)