from datetime import date, timedelta
from io import BytesIO
import flask
import requests
import os
import csv
from datetime import datetime
from flask import Flask, request, send_file,Response,g
import pandas as pd
import json
from psycopg2.extras import RealDictCursor
import psycopg2
from psycopg2.extras import execute_values

try:
    from nepse import Nepse
except ImportError:
    import sys

    sys.path.append("../")
    from nepse import Nepse

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True

# Database connection configuration
app.config['DB_HOST'] = 'localhost'
app.config['DB_NAME'] = 'brokerAnalysis'
app.config['DB_USER'] = 'gaurav'
app.config['DB_PASSWORD'] = 'gaurav'
app.config['DB_PORT'] = 5432

def get_db_connection():
    """Get a connection to the database."""
    if 'db' not in g:
        g.db = psycopg2.connect(
            host=app.config['DB_HOST'],
            database=app.config['DB_NAME'],
            user=app.config['DB_USER'],
            password=app.config['DB_PASSWORD'],
            port=app.config['DB_PORT'],
            cursor_factory=RealDictCursor  # to return dicts from queries
        )
    return g.db

@app.teardown_appcontext
def close_db_connection(error):
    """Close the database connection after each request."""
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()


nepse = Nepse()
nepse.setTLSVerification(False)

routes = {
    "summary": "/summary",
    "scrip-details-ltp":"/scrip-ltp-details",
    "market-open":"/market-open",
    "SupplyDemand": "/SupplyDemand",
    "TopGainers": "/TopGainers",
    "TopLosers": "/TopLosers",
    "TopTenTradeScrips": "/TopTenTradeScrips",
    "TopTenTurnoverScrips": "/TopTenTurnoverScrips",
    "TopTenTransactionScrips": "/TopTenTransactionScrips",
    "IsNepseOpen": "/IsNepseOpen",
    "NepseIndex": "/nepse-index",
    "NepseSubIndices": "/nepse-sub-index",
    "DailyNepseIndexGraph": "/daily-nepse-index-graph",
    "DailyScripPriceGraph": "/charts/scrip",
    "CompanyList": "/CompanyList",
    "SecurityList": "/SecurityList",
    "TradeTurnoverTransactionSubindices": "/TradeTurnoverTransactionSubindices",
    "LiveMarket": "/LiveMarket",
    "Floorsheet":'/Floorsheet',
    "ScriptPastData":"/scrip/history/<string:symbol>",
    "ScriptTexhnicalAnalysis":"/scrip/technical-analysis/<string:symbol>",
    "IndexHistory":"/nepse-index-history/<int:id>",
    "ScripDetails":"/security/<string:symbol>",
    "ScripSubDetails":"/sub-details/security/<string:detail>/<string:symbol>",
    "FileFetch":"/fetch-file",
    'SaveDBAnalysis':'/save-db-analysis'
}

# Helper function for Momentum (10)
def calculate_momentum(df, period=10):
    return df['closePrice'].iloc[-1] - df['closePrice'].shift(period).iloc[-1]

# Helper function for Momentum signal
def get_momentum_signal(momentum):
    if momentum > 0:
        return "Buy"
    elif momentum < 0:
        return "Sell"
    else:
        return "Neutral"

# RSI Calculation Function
def calculate_rsi(df, period=14):
    delta = df['closePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# RSI Signal Function
def get_rsi_signal(rsi):
    if rsi < 30:
        return "Strong Buy"
    elif 30 <= rsi < 40:
        return "Buy"
    elif 40 <= rsi < 60:
        return "Neutral"
    elif 60 <= rsi < 70:
        return "Sell"
    else:
        return "Strong Sell"

# Helper function for Stochastic %K
def calculate_stochastic_k(df, period=14, smooth_k=3):
    # Ensure the DataFrame has enough rows
    if len(df) < period:
        print("Not enough data for calculation.")
        return None  # or return a default value

    # Calculate the rolling lowest low and highest high
    lowest_low = df['lowPrice'].rolling(window=period).min().iloc[-1]
    highest_high = df['highPrice'].rolling(window=period).max().iloc[-1]

    # Debugging output
    print(f"Lowest Low (last {period} days): {lowest_low}")
    print(f"Highest High (last {period} days): {highest_high}")

    # Get the current close price
    current_close = df['closePrice'].iloc[-1]

    # Debugging output
    print(f"Current Close Price: {current_close}")

    # Check for NaN or division by zero
    if pd.isna(lowest_low) or pd.isna(highest_high) or lowest_low == highest_high:
        print("Invalid values for lowest_low or highest_high, cannot compute Stochastic %K.")
        return None  # or return a default value

    # Calculate Stochastic %K
    stochastic_k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)

    # Ensure the calculation is valid
    if pd.isna(stochastic_k):
        print("Stochastic %K calculation resulted in NaN.")
        return None

    # Smooth %K if needed (here it's just the calculated value)
    return stochastic_k

# Stochastic %K Signal Function
def get_stochastic_signal(stochastic_k):
    if pd.isna(stochastic_k):
        return "Neutral"  # Handle NaN case
    if stochastic_k < 20:
        return "Strong Buy"
    elif 20 <= stochastic_k < 30:
        return "Buy"
    elif 30 <= stochastic_k < 70:
        return "Neutral"
    elif 70 <= stochastic_k < 80:
        return "Sell"
    else:
        return "Strong Sell"

# MACD Calculation Function
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['closePrice'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['closePrice'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()

    macd_value = macd.iloc[-1]  # Latest MACD value
    signal_value = macd_signal.iloc[-1]  # Latest MACD Signal value

    macd_signal_result = "Buy" if macd_value > signal_value else "Sell" if macd_value < signal_value else "Neutral"

    return macd_value, macd_signal_result

# Bollinger Bands Calculation Function
def calculate_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['closePrice'].rolling(window).mean()
    rolling_std = df['closePrice'].rolling(window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    current_close = df['closePrice'].iloc[-1]
    bollinger_value = current_close

    # Determine the signal based on the current close price
    if current_close > upper_band.iloc[-1]:
        bollinger_signal = "Sell"
    elif current_close < lower_band.iloc[-1]:
        bollinger_signal = "Buy"
    else:
        bollinger_signal = "Neutral"

    return bollinger_value, bollinger_signal

# Overall Indicator Calculation Function
def calculate_overall_indicator(momentum, rsi, stochastic_k, macd, bollinger):
    signals = []

    # Momentum signal
    if momentum > 0:
        signals.append(1)  # Buy
    elif momentum < 0:
        signals.append(-1)  # Sell
    else:
        signals.append(0)  # Neutral

    # RSI signal
    if rsi < 30:
        signals.append(2)  # Strong Buy
    elif rsi < 40:
        signals.append(1)  # Buy
    elif rsi < 60:
        signals.append(0)  # Neutral
    elif rsi < 70:
        signals.append(-1)  # Sell
    else:
        signals.append(-2)  # Strong Sell

    # Stochastic %K signal
    if pd.isna(stochastic_k):
        signals.append(0)  # Neutral
    elif stochastic_k < 20:
        signals.append(2)  # Strong Buy
    elif stochastic_k < 30:
        signals.append(1)  # Buy
    elif stochastic_k < 70:
        signals.append(0)  # Neutral
    elif stochastic_k < 80:
        signals.append(-1)  # Sell
    else:
        signals.append(-2)  # Strong Sell

    # MACD signal
    if macd > 0:
        signals.append(1)  # Buy
    elif macd < 0:
        signals.append(-1)  # Sell
    else:
        signals.append(0)  # Neutral

    # Bollinger Bands signal
    if bollinger > 0:
        signals.append(1)  # Buy
    elif bollinger < 0:
        signals.append(-1)  # Sell
    else:
        signals.append(0)  # Neutral

    overall_score = sum(signals)

    if overall_score >= 4:
        return "Strong Buy"
    elif overall_score == 3:
        return "Buy"
    elif overall_score == 2:
        return "Neutral"
    elif overall_score == 1:
        return "Sell"
    else:
        return "Strong Sell"

@app.route("/")
def getIndex():
    content = "<BR>".join(
        [f"<a href={value}> {key} </a>" for key, value in routes.items()]
    )
    return f"Serverving hot stock data <BR>{content}"


@app.route(routes["summary"])
def getSummary():
    response = flask.jsonify(_getSummary())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route(routes["FileFetch"])
def getFiles() -> Response:
    try:
        args=request.args
        fetchID=args.get('fetchID')
        # Fetch the PDF from the URL/api/nots/application/fetchFiles?encryptedId={fetchID}",headers=nepse.getAuthorizationHeaders(),verify=False)
        response.raise_for_status()  # Raise an error for bad responses

        # Return the PDF file to the client
        return send_file(
            BytesIO(response.content),
            download_name='file.pdf',
            as_attachment=True,
            mimetype='application/pdf'
        )
    except requests.exceptions.RequestException as e:
        return Response(f"Error fetching PDF: {str(e)}", status=500)


@app.route(routes["Floorsheet"])
def getFloorsheet():
    response=flask.jsonify(nepse.getFloorSheet(show_progress=True))
    save_floorsheet_to_csv(response.data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def _getSummary():
    response = dict()
    for obj in nepse.getSummary(): # type: ignore
        response[obj["detail"]] = obj["value"]
    return response

@app.route(routes["NepseIndex"])
def getNepseIndex():
    response = flask.jsonify(_getNepseIndex())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route(routes["IndexHistory"])
def getNepseHistory(id):
    args=request.args
    limit=args.get('limit',10)
    page = args.get('page', 1)
    response=nepse.requestGETAPI(f"/api/nots/index/history/{id}?limit={limit}&page={page}")
    return flask.jsonify(response)
    # return args


@app.route(routes["ScriptPastData"])
def getScriptPast(symbol):
    args=request.args
    end_date = date.today()
    start_date = end_date - timedelta(days=765)
    response=nepse.getCompanyPriceVolumeHistory(symbol=symbol,start_date=start_date,end_date=end_date)

    # response.headers.add("Access-Control-Allow-Origin", "*")
    return flask.jsonify(response)

@app.route(routes["SaveDBAnalysis"])
def saveDBAnalysis():
    conn = get_db_connection()
    # Get the list of files in the floorsheets directory
    save_dir = "floorsheets"
    files = os.listdir(save_dir)

    # Filter out non-CSV files and sort by date
    csv_files = [f for f in files if f.endswith('.csv')]
    csv_files.sort(reverse=True)

    if not csv_files:
        return flask.jsonify({'error': "No floorsheet files found."}), 404

    # Choose the latest file
    latest_file = csv_files[0]
    latest_file_path = os.path.join(save_dir, latest_file)

    # Read the latest floorsheet data
    df = pd.read_csv(latest_file_path)
    save_broker_share_analysis_to_postgres(df,conn)
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return flask.jsonify({'message':"Success"})

@app.route(routes["ScriptTexhnicalAnalysis"])
def getTA(symbol):
    args = request.args
    interval = args.get('interval', 'daily')  # Default to daily if not specified
    end_date = date.today()
    start_date = end_date - timedelta(days=765)
    # Fetch price data for the company
    response = nepse.getCompanyPriceVolumeHistory(symbol=symbol,start_date=start_date,end_date=end_date)
    data = response['content']
    df = pd.DataFrame(data)

    if len(df) < 14 or 'lowPrice' not in df.columns or 'highPrice' not in df.columns or 'closePrice' not in df.columns:
        return flask.jsonify({"error": "Not enough data for analysis."}), 400

    df['businessDate'] = pd.to_datetime(df['businessDate'])
    df = df.sort_values(by='businessDate')

    # Handle different intervals
    if interval == 'weekly':
        df = df.resample('W', on='businessDate').agg({
            'closePrice': 'last',
            'highPrice': 'max',
            'lowPrice': 'min',
            'totalTradedQuantity': 'sum',
            'totalTradedValue': 'sum',
            'totalTrades': 'sum'
        }).dropna()
    elif interval == 'monthly':
        df = df.resample('M', on='businessDate').agg({
            'closePrice': 'last',
            'highPrice': 'max',
            'lowPrice': 'min',
            'totalTradedQuantity': 'sum',
            'totalTradedValue': 'sum',
            'totalTrades': 'sum'
        }).dropna()

    # Calculate indicators based on the selected interval
    if interval == 'daily':
        df = df.tail(14)  # Get last 14 days
    elif interval == 'monthly':
        df = df.tail(14)  # Get last 14 months

   # Calculate Momentum (10)
    momentum_value = calculate_momentum(df, period=10)
    momentum_signal = get_momentum_signal(momentum_value)

    # Calculate RSI (14)
    rsi_value = calculate_rsi(df, period=14)
    rsi_signal = get_rsi_signal(rsi_value)

    # Calculate Stochastic %K (14, 3)
    stochastic_k_value = calculate_stochastic_k(df, period=14)
    stochastic_k_signal = get_stochastic_signal(stochastic_k_value)

    # Calculate MACD
    macd_value, macd_signal = calculate_macd(df)

    # Calculate Bollinger Bands
    bollinger_value, bollinger_signal = calculate_bollinger_bands(df)


    # Prepare response
    response_data = {
        "indicators": [
            {
                "name": "Momentum (10)",
                "value": int(round(momentum_value)),
                "signal": momentum_signal
            },
            {
                "name": "RSI (14)",
                "value": int(round(rsi_value)), # type:ignore
                "signal": get_rsi_signal(rsi_value)  # Ensure this handles None
            },
            {
                "name": "Stochastic",
                "value": int(round(stochastic_k_value)), #type:ignore
                "signal": get_stochastic_signal(stochastic_k_value)  # Ensure this handles None
            },
             {
                "name": "MACD",
                "signal": macd_signal,
                "value": int(round(macd_value))  # Convert to whole number
            },
            {
                "name": "Bollinger Bands",
                "signal": bollinger_signal,
                "value": int(round(bollinger_value))  # Convert to whole number
            }
        ],
        "overall_indicator": calculate_overall_indicator(momentum_value, rsi_value, stochastic_k_value, macd_value, bollinger_value)  # Pass the values for overall calculation
    }

    return flask.jsonify(response_data)

@app.route(routes["ScripSubDetails"])
def getSubDetails(detail,symbol):
    company_id = nepse.getSecurityIDKeyMap()[symbol]
    print(company_id)
    response=flask.jsonify(nepse.requestGETAPI(f"/api/nots/application/{detail}/{company_id}"))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    # return args

def _getNepseIndex():
    response = dict()
    for obj in nepse.getNepseIndex(): # type: ignore
        response[obj["index"]] = obj
    return response

@app.route(routes["NepseSubIndices"])
def getNepseSubIndices():
    response = flask.jsonify(_getNepseSubIndices())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def _getNepseSubIndices():
    response = []
    reversed_mapper = {
    "Banking SubIndex": "Commercial Banks",
    "Development Bank Index": "Development Banks",
    "Finance Index": "Finance",
    "Hotels And Tourism Index": "Hotels And Tourism",
    "HydroPower Index": "Hydro Power",
    "Investment Index": "Investment",
    "Life Insurance": "Life Insurance",
    "Manufacturing And Processing": "Manufacturing And Processing",
    "Microfinance Index": "Microfinance",
    "Mutual Fund": "Mutual Fund",
    "Non Life Insurance": "Non Life Insurance",
    "Others Index": "Others",
    "Trading Index": "Tradings"
    }
    for obj in nepse.getNepseSubIndices(): # type: ignore
        currentIndex=obj['index']
        if currentIndex in reversed_mapper:
            obj['index'] = reversed_mapper[currentIndex]
        response.append(obj)
    return response


@app.route(routes["TopTenTradeScrips"])
def getTopTenTradeScrips():
    response = flask.jsonify(nepse.getTopTenTradeScrips())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["TopTenTransactionScrips"])
def getTopTenTransactionScrips():
    response = flask.jsonify(nepse.getTopTenTransactionScrips())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["TopTenTurnoverScrips"])
def getTopTenTurnoverScrips():
    response = flask.jsonify(nepse.getTopTenTurnoverScrips())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["SupplyDemand"])
def getSupplyDemand():
    response = flask.jsonify(nepse.getSupplyDemand())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route(routes["ScripDetails"])
def getScripDetails(symbol):
    response=flask.jsonify(nepse.getCompanyDetails(symbol=symbol))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route(routes["DailyScripPriceGraph"])
def getDailyScripPriceGraph():
    args = request.args
    param_scrip_name = args.get("symbol")
    timeframe = args.get("timeframe","1d")
    if param_scrip_name is None:
        raise ValueError("Error: 'symbol' argument is required")
    response = requests.get(f"https://www.onlinekhabar.com/smtm/ticker-page/chart/{param_scrip_name}/{timeframe}")
    responseData=flask.jsonify(response.json())
    responseData.headers.add("Access-Control-Allow-Origin", "*")
    return responseData


@app.route(routes["TopGainers"])
def getTopGainers():
    response = flask.jsonify(nepse.getTopGainers())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["TopLosers"])
def getTopLosers():
    response = flask.jsonify(nepse.getTopLosers())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["market-open"])
def isNepseOpen():
    response = flask.jsonify(nepse.isNepseOpen())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["DailyNepseIndexGraph"])
def getDailyNepseIndexGraph():
    response = flask.jsonify(nepse.getDailyNepseIndexGraph())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response



@app.route(routes["CompanyList"])
def getCompanyList():
    response = flask.jsonify(nepse.getCompanyList())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["SecurityList"])
def getSecurityList():
    response = flask.jsonify(nepse.getSecurityList())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["scrip-details-ltp"])
def getPriceVolume():
    response = flask.jsonify(nepse.getPriceVolume())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["LiveMarket"])
def getLiveMarket():
    response = flask.jsonify(nepse.getLiveMarket())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route(routes["TradeTurnoverTransactionSubindices"])
def getTradeTurnoverTransactionSubindices():
    companies = {company["symbol"]: company for company in nepse.getCompanyList()}
    turnover = {obj["symbol"]: obj for obj in nepse.getTopTenTurnoverScrips()} # type: ignore
    transaction = {obj["symbol"]: obj for obj in nepse.getTopTenTransactionScrips()} # type: ignore
    trade = {obj["symbol"]: obj for obj in nepse.getTopTenTradeScrips()} # type: ignore

    gainers = {obj["symbol"]: obj for obj in nepse.getTopGainers()} # type: ignore
    losers = {obj["symbol"]: obj for obj in nepse.getTopLosers()} # type: ignore

    price_vol_info = {obj["symbol"]: obj for obj in nepse.getPriceVolume()} # type: ignore

    sector_sub_indices = _getNepseSubIndices()
    # this is done since nepse sub indices and sector name are different
    sector_mapper = {
        "Commercial Banks": "Banking SubIndex",
        "Development Banks": "Development Bank Index",
        "Finance": "Finance Index",
        "Hotels And Tourism": "Hotels And Tourism Index",
        "Hydro Power": "HydroPower Index",
        "Investment": "Investment Index",
        "Life Insurance": "Life Insurance",
        "Manufacturing And Processing": "Manufacturing And Processing",
        "Microfinance": "Microfinance Index",
        "Mutual Fund": "Mutual Fund",
        "Non Life Insurance": "Non Life Insurance",
        "Others": "Others Index",
        "Tradings": "Trading Index",
    }

    scrips_details = dict()
    for symbol, company in companies.items():
        company_details = {}

        company_details["symbol"] = symbol
        company_details["sectorName"] = company["sectorName"]
        company_details["totalTurnover"] = (
            turnover[symbol]["turnover"] if symbol in turnover.keys() else 0
        )
        company_details["totalTrades"] = (
            transaction[symbol]["totalTrades"] if symbol in transaction.keys() else 0
        )
        company_details["totalTradeQuantity"] = (
            trade[symbol]["shareTraded"] if symbol in transaction.keys() else 0
        )

        if symbol in gainers.keys():
            (
                company_details["pointChange"],
                company_details["percentageChange"],
                company_details["ltp"],
            ) = (
                gainers[symbol]["pointChange"],
                gainers[symbol]["percentageChange"],
                gainers[symbol]["ltp"],
            )
        elif symbol in losers.keys():
            (
                company_details["pointChange"],
                company_details["percentageChange"],
                company_details["ltp"],
            ) = (
                losers[symbol]["pointChange"],
                losers[symbol]["percentageChange"],
                losers[symbol]["ltp"],
            )
        else:
            (
                company_details["pointChange"],
                company_details["percentageChange"],
                company_details["ltp"],
            ) = (0, 0, 0)

        scrips_details[symbol] = company_details

    sector_details = dict()
    sectors = {company["sectorName"] for company in companies.values()}
    for sector in sectors:
        total_trades, total_trade_quantity, total_turnover = 0, 0, 0
        for scrip_details in scrips_details.values():
            if scrip_details["sectorName"] == sector:
                total_trades += scrip_details["totalTrades"]
                total_trade_quantity += scrip_details["totalTradeQuantity"]
                total_turnover += scrip_details["totalTurnover"]

        sector_details[sector] = {
            "totalTrades": total_trades,
            "totalTradeQuantity": total_trade_quantity,
            "totalTurnover": total_turnover,
            "index": sector_sub_indices[sector_mapper[sector]], #type: ignore
            "sectorName": sector,
        }

    response = flask.jsonify(
        {"scripsDetails": scrips_details, "sectorsDetails": sector_details}
    )

    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def save_broker_share_analysis_to_postgres(df, conn):
    """
    Save broker share analysis data to a PostgreSQL database.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing trade data.
        conn (psycopg2.connection): Connection to the PostgreSQL database.
    """
    try:
        # Calculate total shares bought by each broker per stock and date
        bought_df = df.groupby(['buyerMemberId', 'stockSymbol', 'businessDate'])['contractQuantity'].sum().reset_index()
        bought_df.rename(columns={'buyerMemberId': 'brokerId', 'contractQuantity': 'totalBought'}, inplace=True)

        bought_amount_df=df.groupby(['buyerMemberId','stockSymbol','businessDate'])['contractAmount'].sum().reset_index()
        bought_amount_df.rename(columns={'buyerMemberId':'brokerId','contractAmount':'totalBoughtAmt'},inplace=True)

        bought_avg_df=df.groupby(['buyerMemberId','stockSymbol','businessDate'])['contractAmount'].mean().reset_index()
        bought_avg_df.rename(columns={'buyerMemberId':'brokerId','contractAmount':'avgBoughtRate'},inplace=True)

        # Calculate total shares sold by each broker per stock and date
        sold_df = df.groupby(['sellerMemberId', 'stockSymbol', 'businessDate'])['contractQuantity'].sum().reset_index()
        sold_df.rename(columns={'sellerMemberId': 'brokerId', 'contractQuantity': 'totalSold'}, inplace=True)

        sell_amount_df=df.groupby(['sellerMemberId','stockSymbol','businessDate'])['contractAmount'].sum().reset_index()
        sell_amount_df.rename(columns={'sellerMemberId':'brokerId','contractAmount':'totalSellAmt'},inplace=True)

        sell_avg_df=df.groupby(['sellerMemberId','stockSymbol','businessDate'])['contractAmount'].mean().reset_index()
        sell_avg_df.rename(columns={'sellerMemberId':'brokerId','contractAmount':'avgSoldRate'},inplace=True)

        # Merge the bought and sold DataFrames
        broker_analysis_df = bought_df.merge(
            sold_df, on=['brokerId', 'stockSymbol', 'businessDate'], how='outer'
        ).merge(
            bought_amount_df, on=['brokerId', 'stockSymbol', 'businessDate'], how='outer'
        ).merge(
            sell_amount_df, on=['brokerId', 'stockSymbol', 'businessDate'], how='outer'
        ).merge(
            sell_avg_df, on=['brokerId', 'stockSymbol', 'businessDate'], how='outer'
        ).merge(
            bought_avg_df, on=['brokerId', 'stockSymbol', 'businessDate'], how='outer'
        ).fillna(0)

        # Prepare the insert query
        insert_query = """
        INSERT INTO broker_share_analysis (brokerId, stockSymbol, businessDate, totalBought, totalSold, "totalboughtAmt", "totalsoldAmt", "avgBoughtRate", "avgsoldRate")
        VALUES %s
        """

        # Prepare the data for insertion
        data_tuples = list(broker_analysis_df.itertuples(index=False, name=None))

        # Use execute_values to insert data in bulk
        with conn.cursor() as cursor:
            execute_values(cursor, insert_query, data_tuples)
            conn.commit()

        print("Data successfully inserted into the PostgreSQL database.")

    except Exception as e:
        print("An error occurred while saving data to the database:", e)
        conn.rollback()



def save_floorsheet_to_csv(data):
    """
    Save the floorsheet data to a CSV file with a date-encoded name, including only specified fields.
    Optimized for speed using pandas.
    """
    # Check if the data is bytes and decode it
    if isinstance(data, bytes):
        try:
            data = json.loads(data.decode("utf-8"))  # Decode and parse JSON
        except Exception as e:
            print("Failed to decode and parse data:", e)
            return

    # Debug to inspect the data structure
    if isinstance(data, dict) and "floorsheets" in data:
        data = data["floorsheets"]  # Extract the actual list of records

    # Ensure data is a list
    if not isinstance(data, list):
        print("Unexpected data format! Cannot process.")
        return

    # Define the directory for saving files
    save_dir = "floorsheets"
    os.makedirs(save_dir, exist_ok=True)

    # Generate the file name with the current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(save_dir, f"floorsheet_{date_str}.csv")

    # Define the fields to include in the CSV
    fields = [
        "businessDate",
        "buyerMemberId",
        "contractAmount",
        "contractId",
        "contractQuantity",
        "contractRate",
        "stockSymbol",
        "sellerMemberId",
        "tradeBookId",
        "tradeTime"
    ]

    # Use pandas for faster processing
    try:
        # Filter the records and convert to a DataFrame
        df = pd.DataFrame(data)[fields]

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
    except KeyError as e:
        print(f"Missing field in data: {e}")
    except Exception as e:
        print(f"Error while saving data: {e}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
