

import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Telco Customer Churn Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in ['tenure', 'MonthlyCharges', 'TotalCharges']],
                value='tenure',
                multi=False,
                style={'width': "100%"}
            ),
            dcc.Graph(id='histogram')
        ], width=6),
        
        dbc.Col([
            dcc.Dropdown(
                id='cat-feature-dropdown',
                options=[{'label': col, 'value': col} for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                                                  'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                                                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                                                  'Contract', 'PaperlessBilling', 'PaymentMethod']],
                value='gender',
                multi=False,
                style={'width': "100%"}
            ),
            dcc.Graph(id='countplot')
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='boxplot-tenure', config={'displayModeBar': False})
        ], width=4),
        dbc.Col([
            dcc.Graph(id='boxplot-monthly', config={'displayModeBar': False})
        ], width=4),
        dbc.Col([
            dcc.Graph(id='boxplot-total', config={'displayModeBar': False})
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='contract-tenure-churn', config={'displayModeBar': False})
        ], width=12)
    ])
], fluid=True)


@app.callback(
    Output('histogram', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_histogram(feature):
    fig = px.histogram(df, x=feature, color='Churn', barmode='overlay', nbins=30, title=f'Distribution of {feature}')
    fig.update_layout(xaxis_title=feature, yaxis_title='Count')
    return fig


@app.callback(
    Output('countplot', 'figure'),
    Input('cat-feature-dropdown', 'value')
)
def update_countplot(feature):
    fig = px.histogram(df, x=feature, color='Churn', barmode='group', title=f'Churn Rate by {feature}')
    fig.update_layout(xaxis_title=feature, yaxis_title='Count')
    return fig

@app.callback(
    [Output('boxplot-tenure', 'figure'),
     Output('boxplot-monthly', 'figure'),
     Output('boxplot-total', 'figure')],
    Input('cat-feature-dropdown', 'value')
)
def update_boxplots(_):
    fig_tenure = px.box(df, x='Churn', y='tenure', title='Tenure Distribution by Churn')
    fig_tenure.update_layout(xaxis_title='Churn', yaxis_title='Tenure')
    fig_monthly = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges by Churn')
    fig_monthly.update_layout(xaxis_title='Churn', yaxis_title='Monthly Charges')
    fig_total = px.box(df, x='Churn', y='TotalCharges', title='Total Charges by Churn')
    fig_total.update_layout(xaxis_title='Churn', yaxis_title='Total Charges')
    return fig_tenure, fig_monthly, fig_total


@app.callback(
    Output('contract-tenure-churn', 'figure'),
    Input('cat-feature-dropdown', 'value')
)
def update_contract_tenure_churn(_):
    fig = px.histogram(df, x='tenure', color='Churn', facet_col='Contract', barmode='overlay', nbins=30, 
                       title='Tenure Distribution by Contract Type and Churn')
    fig.update_layout(xaxis_title='Tenure', yaxis_title='Count')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
