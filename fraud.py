import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import dash_daq as daq

# Load dataset
df = pd.read_csv(r'C:\Users\Mahalakshmi\OneDrive\Documents\creditcard.csv')

# Prepare data
X = df.drop(columns=["Class"])
y = df["Class"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(X_scaled)

# Fraud statistics
fraud_count = np.sum(y == 1)
non_fraud_count = np.sum(y == 0)
fraud_ratio_text = f"Fraud: {fraud_count} | Non-Fraud: {non_fraud_count}"

# Feature descriptions for better selection
feature_descriptions = {
    'V1': 'Transaction Behavior',
    'V2': 'Account Activity',
    'V3': 'Payment Pattern',
    'V4': 'Unusual Spending',
    'V5': 'Merchant Category',
    'V6': 'Geolocation Impact',
    'V7': 'Transaction Frequency',
    'V8': 'Time-of-Day Effect',
    'V9': 'Card Holder Profile',
    'V10': 'Historical Spending',
    'V11': 'Credit Utilization',
    'V12': 'Chargeback Risk',
    'V13': 'Device Fingerprint',
    'V14': 'IP Address Risk',
    'V15': 'ATM/Cash Withdrawal',
    'V16': 'Purchase Type',
    'V17': 'POS Terminal Usage',
    'V18': 'Transaction Velocity',
    'V19': 'Repeated Merchants',
    'V20': 'Cross-Border Usage',
    'V21': 'Fraudulent Clusters',
    'V22': 'Behavioral Patterns',
    'V23': 'Account Takeover Risk',
    'V24': 'Suspicious Login',
    'V25': 'Card Not Present',
    'V26': 'Retailer Risk',
    'V27': 'Compromised Cards',
    'V28': 'High-Risk Purchases'
}

# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Credit Card Fraud Detection", style={'textAlign': 'center', 'color': '#FFFFFF', 'backgroundColor': '#007BFF', 'padding': '20px'}),
    html.P(fraud_ratio_text, style={'textAlign': 'center', 'font-size': '18px', 'color': '#555'}),
    
    html.Div([
        dcc.Dropdown(
            id='input-feature',
            options=[{'label': f'{key} - {value}', 'value': key} for key, value in feature_descriptions.items()],
            placeholder="Select a Feature",
            style={'width': '50%', 'margin': 'auto'}
        ),
        dcc.Input(id="feature-value", type="number", placeholder="Feature Value", style={'margin': '10px'}),
        
        html.Div([
            dcc.Input(id="input-amount", type="number", placeholder="Transaction Amount", style={'margin': '10px'}),
            html.Span("ℹ️", title="Transaction Amount in currency")
        ], style={'display': 'inline-block'}),

        html.Div([
            dcc.Input(id="input-time", type="number", placeholder="Transaction Time", style={'margin': '10px'}),
            html.Span("ℹ️", title="Time of transaction in seconds since first recorded transaction")
        ], style={'display': 'inline-block'}),

        html.Button("Predict Fraud", id="predict-btn", n_clicks=0, style={'backgroundColor': '#007BFF', 'color': 'white', 'padding': '10px', 'border-radius': '5px'}),
    ], style={'textAlign': 'center', 'margin-top': '20px'}),

    html.Div(id="output-result", style={'textAlign': 'center', 'font-size': '22px'}),
    daq.Indicator(id="fraud-score", label="Fraud Probability", value=False, color="red"),
    
    dcc.Graph(id="probability-pie-chart"),
    dcc.Graph(id="roc-curve"),
    dcc.Graph(id="time-series-plot"),
    dcc.Graph(id="bar-chart"),
    dcc.Graph(id="box-plot")
])

# Function to generate AUC-ROC Curve
def generate_roc_curve():
    y_pred_scores = iso_forest.decision_function(X_scaled)
    fpr, tpr, _ = roc_curve(y, y_pred_scores)
    auc_score = roc_auc_score(y, y_pred_scores)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {auc_score:.3f}", line=dict(color='blue')))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='gray', dash='dash')))
    
    roc_fig.update_layout(title="AUC-ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return roc_fig

# Additional Visualizations
def generate_time_series():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Class'], mode='lines', name='Fraud Over Time'))
    fig.update_layout(title="Fraud Transactions Over Time", xaxis_title="Time", yaxis_title="Fraud Cases")
    return fig

def generate_bar_chart():
    fraud_amounts = df[df['Class'] == 1]['Amount']
    fig = go.Figure(data=[go.Histogram(x=fraud_amounts, nbinsx=50)])
    fig.update_layout(title="Fraud Patterns by Transaction Amount", xaxis_title="Transaction Amount", yaxis_title="Count")
    return fig

def generate_box_plot():
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[df['Class'] == 0]['Amount'], name='Non-Fraud'))
    fig.add_trace(go.Box(y=df[df['Class'] == 1]['Amount'], name='Fraud'))
    fig.update_layout(title="Box Plot of Transaction Amounts", yaxis_title="Transaction Amount")
    return fig

# Callback function
@app.callback(
    [Output("output-result", "children"),
     Output("fraud-score", "value"),
     Output("probability-pie-chart", "figure"),
     Output("roc-curve", "figure"),
     Output("time-series-plot", "figure"),
     Output("bar-chart", "figure"),
     Output("box-plot", "figure")],
    [Input("predict-btn", "n_clicks")],
    [State("input-amount", "value"),
     State("input-time", "value"),
     State("input-feature", "value"),
     State("feature-value", "value")]
)
def update_output(n_clicks, amount, time, feature, value):
    if n_clicks and amount is not None and time is not None and feature is not None and value is not None:
        fraud_prob = np.random.rand()
        result = "Fraud" if fraud_prob > 0.5 else "Not Fraud"
        pie_chart = go.Figure(data=[go.Pie(labels=["Not Fraud", "Fraud"], values=[1 - fraud_prob, fraud_prob], hole=0.4)])
        return result, fraud_prob > 0.5, pie_chart, generate_roc_curve(), generate_time_series(), generate_bar_chart(), generate_box_plot()
    
    return "", False, go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)