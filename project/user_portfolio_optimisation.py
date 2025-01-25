import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Fetch historical data for given tickers
def fetch_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate daily returns
def calculate_returns(data):
    return data.pct_change().dropna()

def user_defined_portfolio():
    st.title("User-Defined Portfolio")

    # Input for tickers
    st.subheader("Define Your Portfolio")
    tickers = st.text_input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL):", "TCS.NS,ITC.NS,RELIANCE.NS,HDFCBANK.NS,INFY.NS")

    if tickers:
        tickers_list = [ticker.strip().upper() for ticker in tickers.split(",")]
        quantities = []
        prices = []

        st.write("Enter the details for each ticker:")
        for ticker in tickers_list:
            quantity = st.number_input(f"Quantity of {ticker}:", min_value=0.0, value=0.0, step=1.0)
            price = st.number_input(f"Buying Price of {ticker} (₹):", min_value=0.0, value=0.0, step=1.0)
            quantities.append(quantity)
            prices.append(price)

        # Calculate total capital and weights
        investments = [q * p for q, p in zip(quantities, prices)]
        total_capital = sum(investments)
        weights = [(inv / total_capital) * 100 if total_capital > 0 else 0 for inv in investments]

        # Display the portfolio
        portfolio_df = pd.DataFrame({
            "Ticker": tickers_list,
            "Quantity": quantities,
            "Buying Price (₹)": prices,
            "Investment (₹)": investments,
            "Weight (%)": weights
        })

        st.subheader("Portfolio Allocation")
        st.dataframe(portfolio_df)

        st.write(f"Total Capital: ₹{total_capital:.2f}")
# Optimize portfolio weights using mean-variance optimization
def optimize_portfolio(returns):
    """
    Optimizes a portfolio to minimize volatility under the constraint that weights sum to 1.

    Parameters:
        returns (pd.DataFrame): Historical return data of assets (columns as assets, rows as returns).

    Returns:
        dict: Optimized portfolio weights, volatility, and success flag.
    """
    # Calculate the annualized covariance matrix
    cov_matrix = returns.cov() * 252  # Assuming 252 trading days in a year

    # Define the portfolio volatility function
    def portfolio_volatility(weights):
        # Annualized portfolio volatility
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Number of assets
    num_assets = len(returns.columns)

    # Constraints: Weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    # Bounds: Weights must be between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: Equal allocation
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Optimization using Sequential Least Squares Programming (SLSQP)
    result = minimize(
        portfolio_volatility,          # Objective function
        initial_weights,               # Initial guess
        method='SLSQP',                # Optimization method
        bounds=bounds,                 # Weight bounds
        constraints=constraints        # Constraint: sum(weights) = 1
    )

    # Extract results
    optimized_weights = result.x if result.success else None
    optimized_volatility = portfolio_volatility(optimized_weights) if result.success else None

    # Return a dictionary with detailed results
    return {
        'success': result.success,
        'message': result.message,
        'optimized_weights': optimized_weights,
        'optimized_volatility': optimized_volatility
    }


def calculate_var_cvar(returns, weights, confidence_level=0.95):
    portfolio_returns = returns.dot(weights) 
    VaR = np.percentile(portfolio_returns, (1 - confidence_level) )
    CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
    return VaR, CVaR, portfolio_returns  # Return values as percentages

def monte_carlo_simulation(returns, num_portfolios=10000, risk_free_rate=0.03):
    np.random.seed(42)  # For reproducibility
    num_assets = returns.shape[1]
    results = np.zeros((4, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = i
    
    portfolios = pd.DataFrame({
        "Return": results[0],
        "Volatility": results[1],
        "Sharpe": results[2],
    })
    
    max_sharpe_idx = results[2].argmax()
    min_vol_idx = results[1].argmin()
    
    optimal_portfolio = {
        "Return": results[0, max_sharpe_idx],
        "Volatility": results[1, max_sharpe_idx],
        "Sharpe": results[2, max_sharpe_idx],
        "Weights": weights_record[int(results[3, max_sharpe_idx])]
    }
    min_vol_portfolio = {
        "Return": results[0, min_vol_idx],
        "Volatility": results[1, min_vol_idx],
        "Sharpe": results[2, min_vol_idx],
        "Weights": weights_record[int(results[3, min_vol_idx])]
    }
    
    return portfolios, optimal_portfolio, min_vol_portfolio


def plot_efficient_frontier(returns, risk_free_rate=0.03, num_portfolios=5000):
    mean_returns = returns.mean() * 252  # annualized mean returns
    cov_matrix = returns.cov() * 252  # annualized covariance matrix
    num_portfolios = 10000  # Number of random portfolios
    results = np.zeros((4, num_portfolios), dtype=object)  # Store results (Returns, Volatility, Sharpe Ratio, Weights)
    risk_free_rate = 0.03  # Example risk-free rate

    for i in range(num_portfolios):
        # Random portfolio weights
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        
        # Portfolio returns and volatility
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe Ratio
        results[3, i] = weights  # Store weights as a list/array

    # Extract the portfolio with the maximum Sharpe ratio and the minimum volatility
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])

    # Plot the Efficient Frontier
    frontier_fig = go.Figure()

    # Efficient Frontier
    frontier_fig.add_trace(go.Scatter(
        x=results[1],  # Volatility
        y=results[0],  # Return
        mode='markers',
        marker=dict(color=results[2], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio')),
        name='Random Portfolios',
        text=[f"Weights: {np.round(results[3][i], 2)}" for i in range(num_portfolios)],  # Fixed index access
        hovertemplate='<b>Volatility:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<br>%{text}'
    ))

    # Plot the Maximum Sharpe Ratio Portfolio
    frontier_fig.add_trace(go.Scatter(
        x=[results[1, max_sharpe_idx]],
        y=[results[0, max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Maximum Sharpe Ratio Portfolio'
    ))

    # Plot the Minimum Volatility Portfolio
    frontier_fig.add_trace(go.Scatter(
        x=[results[1, min_vol_idx]],
        y=[results[0, min_vol_idx]],
        mode='markers',
        marker=dict(color='blue', size=12),
        name='Minimum Volatility Portfolio'
    ))

    frontier_fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Return',
        template='plotly_white',
        showlegend=True
    )
    return frontier_fig

def asset_clustering(returns, num_clusters=3):
    # Calculate mean returns and volatilities for each asset
    mean_returns = returns.mean() * 252  # Annualized returns
    volatilities = returns.std() * np.sqrt(252)  # Annualized volatilities
    
    # Create a DataFrame of returns and volatilities
    asset_metrics = pd.DataFrame({
        'Return': mean_returns,
        'Volatility': volatilities
    })
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    asset_metrics['Cluster'] = kmeans.fit_predict(asset_metrics[['Return', 'Volatility']])
    
    return asset_metrics, kmeans

# Asset clustering and plotting
def cluster_assets(data, n_clusters=3):
    returns = calculate_returns(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(returns.T)
    clustered_data = pd.DataFrame({'Asset': data.columns, 'Cluster': clusters})

    # Plot clusters
    fig = go.Figure()
    for cluster in range(n_clusters):
        cluster_assets = clustered_data[clustered_data['Cluster'] == cluster]['Asset']
        cluster_returns = returns[cluster_assets]
        avg_return = cluster_returns.mean(axis=1)

        fig.add_trace(go.Scatter(
            x=avg_return.index,
            y=avg_return,
            mode='lines',
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        title='Asset Clustering',
        xaxis_title='Date',
        yaxis_title='Average Returns',
        template='plotly_white'
    )

    return clustered_data, fig


def main():
    st.title("Interactive Portfolio Analysis and Optimization")

   
    
    st.sidebar.subheader("Optimization Hyperparameters")
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (e.g., 0.03 for 3%)", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
    num_portfolios = st.sidebar.number_input("Monte Carlo Portfolios", min_value=100, max_value=100000, value=5000, step=100)
    confidence_level = st.sidebar.slider("VaR/CVaR Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    
    


    # Define the input tab
    st.subheader("Define Your Portfolio")
    tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated, e.g., TCS.NS, ITC.NS, RELIANCE.BO):", "TCS.NS,ITC.NS,RELIANCE.NS,HDFCBANK.NS,INFY.NS")
    
    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        quantities = []
        prices = []

        st.sidebar.subheader("Enter Portfolio Details:")
        total_investments = 0.0

        for ticker in tickers:
            quantity = st.sidebar.number_input(f"Quantity of {ticker}:", min_value=0.0, value=0.0, step=1.0)
            price = st.sidebar.number_input(f"Buying Price of {ticker} (₹):", min_value=0.0, value=0.0, step=1.0)
            quantities.append(quantity)
            prices.append(price)
            total_investments += quantity * price

        if total_investments > 0:
            # Calculate total capital and weights
            investments = [q * p for q, p in zip(quantities, prices)]
            weights = [(inv / total_investments) * 100 for inv in investments]

            # Display portfolio allocation
            portfolio_df = pd.DataFrame({
                "Ticker": tickers,
                "Quantity": quantities,
                "Buying Price (₹)": prices,
                "Investment (₹)": investments,
                "Weight (%)": weights
            })

            st.subheader("Portfolio Allocation")
            st.dataframe(portfolio_df)

            st.write(f"**Total Investment:** ₹{total_investments:.2f}")

            # Fetch historical data
            start_date = st.date_input("Start Date", value=pd.Timestamp("2023-01-01"))
            end_date = st.date_input("End Date", value=pd.Timestamp("today"))
            if st.button("Fetch Data"):
                with st.spinner("Fetching data..."):
                    data = fetch_data(tickers, start_date, end_date)
                if not data.empty:
                    st.success("Data fetched successfully!")
                    st.subheader("Historical Data")
                    st.line_chart(data)

                    # Calculate daily returns
                    returns = calculate_returns(data)

                    # Tabs for different analyses
                    tab1, tab2, tab3 = st.tabs([
                        "Optimal Weights", "Risk Metrics (VaR & CVaR)", "Asset Clustering"
                    ])

                    # Tab 1: Optimal Weights
                    with tab1:
                        st.subheader("Optimized Portfolio Weights")

                        # Dropdown to select optimization method
                        optimization_method = st.selectbox(
                            "Choose Optimization Method:",
                            ["Mean-Variance Optimization", "Monte Carlo Simulation"]
                        )

                        if optimization_method == "Mean-Variance Optimization":
                            # Mean-Variance Optimization
                            result = optimize_portfolio(returns)  # Use the updated portfolio optimization function

                            if result['success']:
                                # Retrieve optimized weights
                                optimal_weights = np.array(result['optimized_weights'], dtype=float)

                                # Calculate optimal investments
                                total_capital = 100000  # Replace with your total investment capital
                                optimal_investments = optimal_weights * total_capital

                                # Display the optimized portfolio as a DataFrame
                                optimized_portfolio_df = pd.DataFrame({
                                    "Ticker": tickers,
                                    "Weight (%)": [round(w * 100, 2) for w in optimal_weights],
                                    "Investment (₹)": [round(inv, 2) for inv in optimal_investments]
                                })
                                st.dataframe(optimized_portfolio_df)

                                # Plot Efficient Frontier
                                frontier_fig = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate, num_portfolios=num_portfolios)
                                st.plotly_chart(frontier_fig)

                            else:
                                # Handle optimization failure
                                st.error(f"Portfolio optimization failed: {result['message']}")


                        elif optimization_method == "Monte Carlo Simulation":
                            try:
                                # Monte Carlo Simulation
                                portfolio_df, optimal_portfolio, min_vol_portfolio = monte_carlo_simulation(
                                    returns, num_portfolios=num_portfolios, risk_free_rate=risk_free_rate
                                )

                                # Debug: Display intermediate data
                                st.write("### Sample Monte Carlo Data")
                                st.write(portfolio_df.head())

                                # Display optimal portfolio
                                st.write("### Optimal Portfolio (Maximum Sharpe Ratio)")
                                st.write(optimal_portfolio)

                                # Display minimum volatility portfolio
                                st.write("### Minimum Volatility Portfolio")
                                st.write(min_vol_portfolio)

                                # Plot Monte Carlo portfolios
                                monte_carlo_fig = go.Figure()

                                monte_carlo_fig.add_trace(go.Scatter(
                                    x=portfolio_df['Volatility'],
                                    y=portfolio_df['Return'],
                                    mode='markers',
                                    marker=dict(color=portfolio_df['Sharpe'], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio')),
                                    name='Portfolios',
                                ))

                                monte_carlo_fig.add_trace(go.Scatter(
                                    x=[optimal_portfolio['Volatility']],
                                    y=[optimal_portfolio['Return']],
                                    mode='markers',
                                    marker=dict(color='red', size=10),
                                    name='Maximum Sharpe Ratio Portfolio'
                                ))

                                monte_carlo_fig.add_trace(go.Scatter(
                                    x=[min_vol_portfolio['Volatility']],
                                    y=[min_vol_portfolio['Return']],
                                    mode='markers',
                                    marker=dict(color='blue', size=10),
                                    name='Minimum Volatility Portfolio'
                                ))

                                monte_carlo_fig.update_layout(
                                    title='Monte Carlo Simulation - Portfolio Optimization',
                                    xaxis_title='Volatility (Risk)',
                                    yaxis_title='Return',
                                    template='plotly_white'
                                )

                                st.plotly_chart(monte_carlo_fig)

                            except Exception as e:
                                st.error(f"Error during Monte Carlo Simulation: {e}")


                    # Tab 2: Risk Metrics (VaR & CVaR)
                    with tab2:
                        st.subheader("Risk Metrics")
                        VaR, CVaR, portfolio_returns = calculate_var_cvar(returns, weights, confidence_level=confidence_level)
                        st.metric("VaR (95%)", f"{VaR:.4f}%")
                        st.metric("CVaR (95%)", f"{CVaR:.4f}%")

                        # Portfolio returns distribution
                        risk_fig = go.Figure()
                        risk_fig.add_trace(go.Histogram(
                            x=portfolio_returns, nbinsx=50,
                            marker=dict(color='skyblue', line=dict(color='black', width=1))
                        ))
                        risk_fig.add_trace(go.Scatter(
                            x=[VaR, VaR], y=[0, 50],
                            mode='lines', line=dict(color='red', dash='dash'), name=f'VaR ({VaR:.4f})'
                        ))
                        risk_fig.add_trace(go.Scatter(
                            x=[CVaR, CVaR], y=[0, 50],
                            mode='lines', line=dict(color='orange', dash='dash'), name=f'CVaR ({CVaR:.4f})'
                        ))

                        risk_fig.update_layout(
                            title='Portfolio Returns Distribution with VaR and CVaR',
                            xaxis_title='Daily Returns', yaxis_title='Frequency', barmode='overlay'
                        )
                        st.plotly_chart(risk_fig)

                        # Tab 3: Asset Clustering
                        with tab3:
                            st.subheader("Asset Clustering")
                            asset_metrics, kmeans = asset_clustering(returns)
                            st.write("### Clusters based on Return and Volatility")
                            st.dataframe(asset_metrics)

                                # Plot clustering results
                            cluster_fig = go.Figure()
                            for cluster in range(kmeans.n_clusters):
                                    cluster_data = asset_metrics[asset_metrics['Cluster'] == cluster]
                                    cluster_fig.add_trace(go.Scatter(
                                        x=cluster_data['Volatility'],
                                        y=cluster_data['Return'],
                                        mode='markers',
                                        marker=dict(size=10),
                                        name=f'Cluster {cluster}'
                                    ))
                            cluster_fig.update_layout(
                                title="Asset Clustering Based on Return and Volatility",
                                xaxis_title="Volatility (Risk)",
                                yaxis_title="Return",
                                template="plotly_white"
                                )
                            st.plotly_chart(cluster_fig)
                            clustered_data, cluster_fig = cluster_assets(data)

                            st.write("Clustered Assets:")
                            st.dataframe(clustered_data)
                            st.plotly_chart(cluster_fig, use_container_width=True)

        else:
            st.warning("Please enter valid quantities and prices for the portfolio.")
    else:
        st.info("Please enter tickers to begin.")

if __name__ == "__main__":
    main()
