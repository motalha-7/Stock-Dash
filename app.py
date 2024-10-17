import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime as dt
import joblib
import os
from model import train_svr_model, load_svr_model

def load_svr_model():
    return joblib.load('svr_model.pkl')




app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        # Navigation Area
        html.P("Welcome to the Stock Dash App!", className="start"),
        dcc.Input(id="stock-code-input", type="text", placeholder="Enter stock code"),
        dcc.DatePickerRange(
            id='date-range-picker',
            start_date=dt(2023, 1, 1),
            end_date=dt.now(),
            display_format='YYYY-MM-DD'
        ),
        html.Button("Get Stock Price", id="get-stock-price"),
        html.Button("Show Indicators", id="show-indicators"),
        dcc.Input(id='forecast-days', type='number', placeholder='Days to forecast'),
        html.Button('Forecast', id='forecast-button'),
    ], className="nav"),
    
    # Content Area
    html.Div([
        html.Div([
            html.Img(id="company-logo", className="logo"),
            html.H1(id="company-name", className="company-name"),
        ], className="header"),
        html.Div(id="description", className="description_ticker"),
        html.Div(dcc.Graph(id="stock-price-graph"), className="graph-container"),
        html.Div(dcc.Graph(id="indicator-graph"), className="graph-container"),
        html.Div( dcc.Graph(id='forecast-graph'), className='graph-container' ),
    ], className="content")
], className="container")

@app.callback(
    [Output("description", "children"),
     Output("company-logo", "src"),
     Output("company-name", "children")],
    [Input("stock-code-input", "value")]
)
def update_company_info(stock_code):
  
    if stock_code is None:
        return "Enter a valid stock code", "/assets/logo.png", "PEAK PIVOT"

    stock = yf.Ticker(stock_code)
    company_info = stock.info
    description = company_info.get('longBusinessSummary', 'No description available.')
    logo = company_info.get('logo_url', '/assets/logo.png')
    name = company_info.get('longName', 'PEAK PIVOT')

    return description, logo, name

@app.callback(
    Output("stock-price-graph", "figure"),
    [Input("get-stock-price", "n_clicks")],
    [State("stock-code-input", "value"),
     State("date-range-picker", "start_date"),
     State("date-range-picker", "end_date")]
)
def update_stock_graph(n_clicks, stock_code, start_date, end_date):
    if not n_clicks or stock_code is None:
        return go.Figure()

    df = yf.download(stock_code, start=start_date, end=end_date)

    figure = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    figure.update_layout(title=f'Stock Price for {stock_code}')

    return figure

@app.callback(
    Output("indicator-graph", "figure"),
    [Input("show-indicators", "n_clicks")],
    [State("stock-code-input", "value"),
     State("date-range-picker", "start_date"),
     State("date-range-picker", "end_date")]
)
def update_indicator_graph(n_clicks, stock_code, start_date, end_date):
    if not n_clicks or stock_code is None:
        return go.Figure()

    df = yf.download(stock_code, start=start_date, end=end_date)
    df['SMA'] = df['Close'].rolling(window=20).mean()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    figure.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='20-day SMA'))
    figure.update_layout(title=f'Indicators for {stock_code}')

    return figure


@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-button', 'n_clicks')],
    [State('stock-code-input', 'value'),
     State('forecast-days', 'value')]
)
def update_forecast_graph(n_clicks, stock_code, forecast_days):
    if not n_clicks or stock_code is None or forecast_days is None:
        return go.Figure()
    
    # Fetch the last 60 days of stock data
    df = yf.download(stock_code, period='1mo')
    
    # Check if the dataframe is empty
    if df.empty:
        return go.Figure().update_layout(
            title=f'No data available for {stock_code}',
            xaxis_title='Date',
            yaxis_title='Price'
        )

    # Calculate the number of days since the first day in the dataset
    df['Day'] = (df.index - df.index[0]).days
    X = np.array(df['Day']).reshape(-1, 1)
    
    # Load the SVR model
    model = load_svr_model()

    # Predict future values for the specified forecast days
    future_days = np.arange(X[-1, 0] + 1, X[-1, 0] + forecast_days + 1).reshape(-1, 1)
    forecast = model.predict(future_days)

    # Combine the actual dates with the forecasted dates
    last_date = df.index[-1]
    future_dates = [last_date + pd.Timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
    
    # Create the figure with actual and forecasted data
    figure = go.Figure()

    # Add actual close price line
    figure.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        mode='lines', 
        name='Actual Close Price',
        line=dict(color='blue', width=2)
    ))

    # Add forecasted price line
    figure.add_trace(go.Scatter(
        x=future_dates, 
        y=forecast, 
        mode='lines', 
        name='Forecasted Price',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Update the layout for better visualization
    figure.update_layout(
        title=f'{forecast_days}-Day Forecast for {stock_code}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis=dict(showgrid=True, zeroline=False, showline=True, showticklabels=True),
        yaxis=dict(showgrid=True, zeroline=False, showline=True, showticklabels=True),
        hovermode="x"
    )

    # Optional: Add a vertical line to mark the start of the forecast
    figure.add_shape(
        dict(
            type="line",
            x0=last_date,
            y0=0,
            x1=last_date,
            y1=forecast.max(),
            line=dict(color='red', dash='dash')
        )
    )

    return figure





if __name__ == '__main__':
    app.run_server(debug=True)
