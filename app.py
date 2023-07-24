import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from markup import real_estate_app, real_estate_app_hf
import feedparser

API_URL = "https://api.coingecko.com/api/v3"

PASSWORD = 'Ethan101'

def authenticate(password):
    return password == PASSWORD

def get_ethereum_data():
    response = requests.get(f"{API_URL}/coins/markets", params={"ids": "ethereum", "vs_currency": "usd"})
    data = response.json()
    return data

def format_price(price):
    return "{:.10f}".format(price)

def get_new_tokens():
    response = requests.get(f"{API_URL}/coins/ethereum/market_chart", params={"vs_currency": "usd", "days": 1})
    data = response.json()
    
    new_tokens = []
    for token in data["market_caps"]:
        timestamp, market_cap = token
        if market_cap > 20000:
            coin_token = data["prices"][data["market_caps"].index(token)][1]
            # Convert the coin_token to hex format
            coin_token_hex = float_to_hex(coin_token)
            new_tokens.append((coin_token_hex, timestamp, market_cap))
    return new_tokens

def float_to_hex(f):
    # Convert the float to its hexadecimal representation
    _, hex_representation = f.hex().split('x')
    return "0x" + hex_representation

def predict_price(df_price_history, days):
    X = df_price_history.index.values.reshape(-1, 1)
    y = df_price_history["price"].values
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    last_date = df_price_history.iloc[-1]["date"]
    lr_future_dates = pd.date_range(last_date, periods=days+1)[1:]
    lr_future_predictions = lr_model.predict(np.array(range(1, days+1)).reshape(-1, 1))
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X, y)
    rf_future_dates = pd.date_range(last_date, periods=days+1)[1:]
    rf_future_predictions = rf_model.predict(np.array(range(1, days+1)).reshape(-1, 1))

    return lr_future_dates, lr_future_predictions, rf_future_dates, rf_future_predictions

def tab1():
    st.header("ethereum Cryptocurrency Predictions Demo")  
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("Hotpot.png", use_column_width=True)
    with col2:
        st.markdown(real_estate_app(), unsafe_allow_html=True)
    st.markdown(real_estate_app_hf(),unsafe_allow_html=True) 


    github_link = '[<img src="https://badgen.net/badge/icon/github?icon=github&label">](https://github.com/ethanrom)'
    #huggingface_link = '[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/ethanrom)'

    st.write(github_link + '&nbsp;&nbsp;&nbsp;', unsafe_allow_html=True)

def tab2():
    ethereum_data = get_ethereum_data()
    if ethereum_data:
        ethereum_info = {
            "Symbol": ethereum_data[0]["symbol"],
            "Current Price": format_price(ethereum_data[0]["current_price"]),
            "Market Cap": ethereum_data[0]["market_cap"],
            "Total Volume": ethereum_data[0]["total_volume"],
            "Circulating Supply": ethereum_data[0]["circulating_supply"],
        }
        df_ethereum = pd.DataFrame(ethereum_info, index=[0])
        
        st.markdown("## Ethereum Information")
        st.dataframe(df_ethereum)
        
        # Visualize Market Cap and Total Volume
        market_cap = ethereum_data[0]["market_cap"]
        total_volume = ethereum_data[0]["total_volume"]
        df_market_cap_volume = pd.DataFrame({"Metric": ["Market Cap", "Total Volume"],
                                             "Value (USD)": [market_cap, total_volume]})
        st.markdown("## Market Cap and Total Volume")
        st.bar_chart(df_market_cap_volume, x="Metric", y="Value (USD)")
        st.markdown(
            """
            The bar chart above shows the current market capitalization and total trading volume of Ethereum in USD.
            """
        )
        
        circulating_supply = ethereum_data[0]["circulating_supply"]
        max_supply = ethereum_data[0]["total_supply"]
        
        st.markdown("## Supply Information")
        st.write(f"**Circulating Supply:** {circulating_supply:.2f} Ethereum")
        st.write(f"**Max Supply:** {max_supply:.2f} Ethereum")

        # Additional Visualization: Pie Chart for Circulating vs. Max Supply
        supply_data = pd.DataFrame({
            "Supply": ["Circulating Supply", "Max Supply"],
            "Amount (Ethereum)": [circulating_supply, max_supply]
        })
        fig = px.pie(supply_data, values="Amount (Ethereum)", names="Supply", title="Circulating vs. Max Supply")
        st.markdown("## Circulating vs. Max Supply")
        st.write(
            """
            The pie chart above compares the circulating supply and maximum supply of Ethereum in terms of the number of tokens.
            """
        )
        st.plotly_chart(fig)

        # Show new Ethereum tokens created in the last 24 hours with market cap > $20,000
        new_tokens = get_new_tokens()
        if new_tokens:
            st.markdown("## New Ethereum Tokens Created in the Last 24 Hours (Market Cap > $20,000)")
            for coin_token, timestamp, market_cap in new_tokens:
                time_created = pd.to_datetime(timestamp, unit="ms").strftime("%H:%M:%S")
                st.write(f"COIN TOKEN: {coin_token}, TIME CREATED: {time_created}, MARKET CAP: ${market_cap/1e6:.1f}MM")
        else:
            st.write("No new Ethereum tokens with market cap > $20,000 created in the last 24 hours.")


def tab3():
    ethereum_data = get_ethereum_data()
    if ethereum_data:
        response = requests.get(f"{API_URL}/coins/ethereum/market_chart", params={"vs_currency": "usd", "days": "30"})
        price_history = response.json()
        df_price_history = pd.DataFrame(price_history["prices"], columns=["date", "price"])
        df_price_history["date"] = pd.to_datetime(df_price_history["date"], unit="ms")
        
        st.markdown("## ethereum Price History")
        fig = px.line(df_price_history, x="date", y="price", title="ethereum Price History")
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)
        st.markdown(
            """
            The line chart above shows the historical price trend of ethereum over the last 30 days.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
        
            price_stats = df_price_history["price"].describe()
            st.markdown("## Price Statistics")
            st.write(price_stats)
        
        with col2:
            st.markdown("## Price Distribution")
            fig_hist = px.histogram(df_price_history, x="price", nbins=20, title="Histogram of Price Distribution")
            fig_hist.update_layout(xaxis_title="Price (USD)", yaxis_title="Count")
            st.plotly_chart(fig_hist)
        st.markdown(
            """
            The histogram above displays the distribution of ethereum prices over the last 30 days.
            """
        )

    else:
        st.write("Failed to retrieve ethereum data")



def tab4():
    ethereum_data = get_ethereum_data()
    if ethereum_data:
        response = requests.get(f"{API_URL}/coins/ethereum/market_chart", params={"vs_currency": "usd", "days": "30"})
        price_history = response.json()
        df_price_history = pd.DataFrame(price_history["prices"], columns=["date", "price"])
        df_price_history["date"] = pd.to_datetime(df_price_history["date"], unit="ms")

        # Perform predictions
        days = 30
        lr_future_dates, lr_future_predictions, rf_future_dates, rf_future_predictions = predict_price(df_price_history, days)

        # Visualize predictions using line charts
        st.markdown("## Price Predictions")
        st.subheader("Linear Regression Prediction")
        df_lr_predicted = pd.DataFrame({"Date": lr_future_dates, "Predicted Price": lr_future_predictions})
        st.line_chart(df_lr_predicted, x="Date", y="Predicted Price")

        st.subheader("Random Forest Regression Prediction")
        df_rf_predicted = pd.DataFrame({"Date": rf_future_dates, "Predicted Price": rf_future_predictions})
        st.line_chart(df_rf_predicted, x="Date", y="Predicted Price")

        # Additional Visualization: Combined Line Chart for Actual and Predicted Prices
        df_combined = pd.concat([df_price_history, df_lr_predicted.rename(columns={"Predicted Price": "price"})])
        df_combined["Type"] = ["Actual"] * len(df_price_history) + ["Predicted (LR)"] * len(df_lr_predicted)
        fig_combined = px.line(df_combined, x="date", y="price", color="Type", title="Actual vs. Predicted (LR) Prices")
        fig_combined.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig_combined)

        # Add text explanation for predictions
        st.markdown("## Predictions Explanation")
        st.write(
            """
            The price predictions are estimated using regression models: Linear Regression (LR) and Random Forest Regression (RF).
            The line charts show the predicted prices over the next 30 days based on historical price data.
            """
        )

    else:
        st.write("Failed to retrieve ethereum data")

#tab5
RSS_FEED_URLS = {
    "CryptoNews": "https://cryptonews.com/news/feed/",
    "CoinDesk": "https://www.coindesk.com/feed",
    "CryptoSlate": "https://cryptoslate.com/feed/",
    # Add more RSS feed URLs here
}

def fetch_latest_news(url):
    feed = feedparser.parse(url)
    entries = feed.entries[:5]  # Fetching the latest 5 news entries
    return entries    

def filter_news_by_keyword(entries, keyword):
    filtered_entries = []
    for entry in entries:
        if keyword.lower() in entry.title.lower() or keyword.lower() in entry.summary.lower():
            filtered_entries.append(entry)
    return filtered_entries

def display_news_entry(entry):
    st.markdown(f"## {entry.title}")
    st.write(entry.summary)
    st.write(f"Published on: {entry.published}")
    st.write("---")

def tab5():
    selected_feeds = st.multiselect("Select RSS Feeds", list(RSS_FEED_URLS.keys()), default=["CryptoNews"])
    filter_keyword = st.text_input("Filter by keyword (e.g., ethereum)")

    for feed in selected_feeds:
        st.markdown(f"### {feed} News")

        if feed in RSS_FEED_URLS:
            entries = fetch_latest_news(RSS_FEED_URLS[feed])

            if filter_keyword:
                entries = filter_news_by_keyword(entries, filter_keyword)

            for entry in entries:
                display_news_entry(entry)
        else:
            st.write(f"No RSS feed URL found for {feed}")

def tab6():
    st.header("Download script")
    st.markdown(
        """
        download the standalone python script to print new tokens
        """
    )
    st.image("eth.PNG")
    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):
        # Contents of the get_new_coins.py file
        script_content = """
import requests
import pandas as pd

API_URL = "https://api.coingecko.com/api/v3"

def float_to_hex(f):
    _, hex_representation = f.hex().split('x')
    return "0x" + hex_representation

def get_new_tokens():
    response = requests.get(f"{API_URL}/coins/ethereum/market_chart", params={"vs_currency": "usd", "days": 1})
    data = response.json()
    
    new_tokens = []
    for token in data["market_caps"]:
        timestamp, market_cap = token
        if market_cap > 20000:
            coin_token = data["prices"][data["market_caps"].index(token)][1]
            coin_token_hex = float_to_hex(coin_token)
            new_tokens.append((coin_token_hex, timestamp, market_cap))
    return new_tokens

if __name__ == "__main__":
    new_tokens = get_new_tokens()
    if new_tokens:
        print("New Ethereum Tokens Created in the Last 24 Hours (Market Cap > $20,000)")
        for coin_token, timestamp, market_cap in new_tokens:
            time_created = pd.to_datetime(timestamp, unit="ms").strftime("%H:%M:%S")
            print(f"COIN: {coin_token}, TIME CREATED: {time_created}, MARKET CAP: ${market_cap/1e6:.1f}MM")
    else:
        print("No new Ethereum tokens with market cap > $20,000 created in the last 24 hours.")
"""

        # Display the content of the script in the app
        st.code(script_content, language="python")

        # Download link for the script
        file_name = "get_new_coins.py"
        st.download_button(
            label="Download get_new_coins.py",
            data=script_content,
            file_name=file_name,
            mime="text/plain",
        )
    else:
        # Password is incorrect, show an error message
        st.error('Invalid password. Access denied.')

def main():
    st.set_page_config(page_title="ethereum Dashboard", page_icon=":memo:", layout="wide")
    tabs = ["Intro", "ethereum Information", "ethereum Price History", "Price Predictions", "News", "Download Script"]

    with st.sidebar:

        current_tab = option_menu("Select a Tab", tabs, menu_icon="cast")

    tab_functions = {
    "Intro": tab1,
    "ethereum Information": tab2,
    "ethereum Price History": tab3,
    "Price Predictions": tab4,
    "News": tab5,
    "Download Script": tab6,
    }

    if current_tab in tab_functions:
        tab_functions[current_tab]()



if __name__ == "__main__":
    main()