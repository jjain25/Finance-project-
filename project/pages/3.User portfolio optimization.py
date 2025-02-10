import streamlit as st 
from user_portfolio_optimisation import *


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
