import streamlit as st 
from portfolio_optimisation import *

def main():
    st.title("Portfolio Optimization and Risk Analysis")
    st.write("This application helps analyze and optimize a portfolio using Monte Carlo simulation, Mean-Variance optimization, and clustering techniques.")

    # Sidebar inputs
    st.sidebar.header("User Input")
    tickers = st.sidebar.text_input("Enter asset tickers (comma-separated):", 
                                     "TCS.NS,ITC.NS,RELIANCE.NS,HDFCBANK.NS,INFY.NS")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (in decimal form):", value=0.03, step=0.01)

    # Hyperparameters for optimization methods
    num_simulations = st.sidebar.number_input("Number of Monte Carlo Simulations:", min_value=1000, max_value=10000, value=5000, step=500)
    risk_tolerance = st.sidebar.number_input("Risk Tolerance (for Mean-Variance Optimization):", value=0.5, step=0.1)

    # Process inputs
    tickers = tickers.split(',')
    data = fetch_data(tickers, start_date, end_date)
    returns = calculate_returns(data)

    # Display historical data
    st.write("Historical Adjusted Close Prices:")
    st.line_chart(data)

    # Display correlation matrix
    st.subheader("Correlation Matrix")
    correlation_fig = plot_correlation_matrix(returns)
    st.plotly_chart(correlation_fig)
    st.write("""
    **Analysis of Correlation Matrix:**  
    - This heatmap shows the relationships between the returns of different assets.  
    - Positive values (closer to 1) indicate that two assets tend to move in the same direction.  
    - Negative values (closer to -1) indicate that two assets move in opposite directions.  
    - A diversified portfolio typically includes assets with low or negative correlations to minimize overall risk.
    """)
    
    # User selection for method
    method = st.sidebar.selectbox("Select Optimization Method", ["Monte Carlo Simulation", "Mean-Variance Optimization"])

    if st.sidebar.button("Run Analysis"):
        # Tabs for different outputs
        tabs = st.tabs(["Portfolio Optimization", "Risk Metrics", "Asset Clustering"])

        # Tab 1: Portfolio Optimization

        with tabs[0]:
            st.subheader("Optimized Portfolio")
            if method == "Monte Carlo Simulation":
                #def monte_carlo_simulation(returns, num_portfolios=5000, risk_free_rate=0.03):
                portfolio_df, optimal_portfolio, min_vol_portfolio = monte_carlo_simulation(returns,num_simulations, risk_free_rate=risk_free_rate)
                weights = optimal_portfolio['Weights']
                
                st.write("### Optimal Portfolio Weights")
                st.dataframe(pd.DataFrame({'Ticker': tickers, 'Weight': weights}))
                st.write("**Expected Return:**", f"{optimal_portfolio['Return']:.2%}")
                st.write("**Volatility (Risk):**", f"{optimal_portfolio['Volatility']:.2%}")
                st.write("**Sharpe Ratio:**", f"{optimal_portfolio['Sharpe']:.2f}")

                st.write("### Minimum Volatility Portfolio")
                st.dataframe(pd.DataFrame({'Ticker': tickers, "Weights": min_vol_portfolio["Weights"]}))
                st.write("**Expected Return:**", f"{min_vol_portfolio['Return']:.2%}")
                st.write("**Volatility (Risk):**", f"{min_vol_portfolio['Volatility']:.2%}")
                st.write("**Sharpe Ratio:**", f"{min_vol_portfolio['Sharpe']:.2f}")

                # Plot Efficient Frontier
                frontier_fig = plot_efficient_frontier(returns)
                st.plotly_chart(frontier_fig)
                st.write("""
        **Analysis:**  
        - The Maximum Sharpe Ratio portfolio provides the best trade-off between risk and return.  
          - It maximizes the excess return per unit of risk, making it a balanced choice for growth-oriented investors.  
          - This portfolio often includes assets with strong historical performance and diversification benefits.  

        - The Minimum Volatility portfolio is ideal for risk-averse investors but may have lower returns.  
          - It aims to reduce portfolio fluctuations, offering stability during volatile market conditions.  
          - This portfolio typically emphasizes low-risk assets like bonds or defensive equities.  

        - The Efficient Frontier highlights the portfolios that offer the best possible return for a given level of risk.  
          - Portfolios along the Efficient Frontier are optimized for the highest return at each risk level.  
          - Moving up the frontier increases returns but also entails higher risk.  

        - Diversification plays a critical role in achieving these portfolios.  
          - Combining uncorrelated assets reduces overall portfolio risk.  
          - Proper asset allocation ensures resilience against market downturns.  

        - Consideration of individual risk tolerance and investment horizon is essential.  
          - Aggressive investors may favor portfolios closer to the Maximum Sharpe Ratio.  
          - Conservative investors may lean towards the Minimum Volatility portfolio.  

        - Regular portfolio rebalancing ensures alignment with investment goals.  
          - Market changes can shift the risk-return profile, necessitating adjustments.  
          - Rebalancing helps maintain the desired level of diversification and risk.  
        """)


            elif method == "Mean-Variance Optimization":
                weights = optimize_portfolio(returns,risk_free_rate)
                st.write("### Optimal Portfolio Weights")
                st.dataframe(pd.DataFrame({'Ticker': tickers, 'Weight': weights}))

                # Plot Efficient Frontier
                frontier_fig = plot_efficient_frontier(returns)
                st.plotly_chart(frontier_fig)
                st.write("""
        **Analysis:**  
        - The Maximum Sharpe Ratio portfolio provides the best trade-off between risk and return.  
          - It maximizes the excess return per unit of risk, making it a balanced choice for growth-oriented investors.  
          - This portfolio often includes assets with strong historical performance and diversification benefits.  

        - The Minimum Volatility portfolio is ideal for risk-averse investors but may have lower returns.  
          - It aims to reduce portfolio fluctuations, offering stability during volatile market conditions.  
          - This portfolio typically emphasizes low-risk assets like bonds or defensive equities.  

        - The Efficient Frontier highlights the portfolios that offer the best possible return for a given level of risk.  
          - Portfolios along the Efficient Frontier are optimized for the highest return at each risk level.  
          - Moving up the frontier increases returns but also entails higher risk.  

        - Diversification plays a critical role in achieving these portfolios.  
          - Combining uncorrelated assets reduces overall portfolio risk.  
          - Proper asset allocation ensures resilience against market downturns.  

        - Consideration of individual risk tolerance and investment horizon is essential.  
          - Aggressive investors may favor portfolios closer to the Maximum Sharpe Ratio.  
          - Conservative investors may lean towards the Minimum Volatility portfolio.  

        - Regular portfolio rebalancing ensures alignment with investment goals.  
          - Market changes can shift the risk-return profile, necessitating adjustments.  
          - Rebalancing helps maintain the desired level of diversification and risk.  
        """)


        # Tab 2: Risk Metrics
        with tabs[1]:
            st.subheader("Risk Metrics")
            VaR, CVaR, portfolio_returns = calculate_var_cvar(returns, weights)
            st.write(f"**VaR (95%):** {VaR:.4f}")
            st.write(f"**CVaR (95%):** {CVaR:.4f}")

            # Portfolio returns distribution
            hist_data = go.Histogram(
                x=portfolio_returns,
                nbinsx=50,
                marker=dict(color='skyblue', line=dict(color='black', width=1)),
                name='Portfolio Returns'
            )

            VaR_line = go.Scatter(
                x=[VaR, VaR],
                y=[0, 50],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'VaR ({VaR:.4f})'
            )

            CVaR_line = go.Scatter(
                x=[CVaR, CVaR],
                y=[0, 50],
                mode='lines',
                line=dict(color='orange', dash='dash'),
                name=f'CVaR ({CVaR:.4f})'
            )

            layout = go.Layout(
                title='Portfolio Returns Distribution with VaR and CVaR',
                xaxis=dict(title='Daily Returns'),
                yaxis=dict(title='Frequency'),
                barmode='overlay'
            )

            risk_fig = go.Figure(data=[hist_data, VaR_line, CVaR_line], layout=layout)
            st.plotly_chart(risk_fig)
            st.write("""
    **Analysis of VaR and CVaR Plot:**  
    - **VaR (Value at Risk):**  
        The VaR metric quantifies the maximum potential loss over a specified time horizon at a given confidence level.  
        For instance, a VaR of -0.03 with a 95% confidence level implies that there is only a 5% probability of the portfolio losing more than 3% of its value in a single day.  
        VaR is useful for setting risk limits and understanding downside risk but does not provide information about the magnitude of losses beyond the VaR threshold.  

    - **CVaR (Conditional Value at Risk):**  
        CVaR, also known as Expected Shortfall, measures the average loss in scenarios where losses exceed the VaR threshold.  
        For example, a CVaR of -0.05 means that if losses exceed the VaR (i.e., fall in the worst 5% of outcomes), the average loss would be 5% of the portfolio's value.  
        CVaR offers a more complete picture of tail risk compared to VaR, making it a preferred metric for extreme risk analysis.  

    - **Insights from the Plot:**  
        - The **histogram** displays the distribution of portfolio returns, helping to identify the likelihood of different return levels.  
        - The **VaR and CVaR thresholds** are clearly marked as vertical lines on the plot, providing a visual representation of risk boundaries.  
        - By examining the plot, users can assess whether the portfolio's return distribution has fat tails or is skewed, which can significantly impact risk assessments.  
        - This analysis aids in decision-making, such as optimizing portfolio allocation, adjusting hedging strategies, or defining risk limits.

    - **Additional Considerations:**  
        - **Time Horizon:** VaR and CVaR values depend on the time period chosen (e.g., daily, weekly). The interpretation and applicability vary accordingly.  
        - **Confidence Levels:** The confidence level (e.g., 95%, 99%) affects the stringency of risk thresholds. Higher confidence levels result in more conservative estimates.  
        - **Model Assumptions:** The accuracy of VaR and CVaR depends on the assumptions of the return distribution (e.g., normality). Real-world return distributions often exhibit non-normal characteristics such as skewness and kurtosis.  
        - **Complementary Metrics:** While VaR and CVaR are important, combining them with other risk measures (e.g., stress testing, Sharpe ratio) provides a more holistic view of portfolio risk.
    """)


        # Tab 3: Asset Clustering
        with tabs[2]:
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
            st.write("""
        **Analysis:**  
        - Assets within the same cluster tend to have similar return patterns.  
        - This clustering can help diversify your portfolio by selecting assets from different clusters.  
        - Visualizing the clusters over time provides insights into the relative performance of asset groups.  
- **Cluster Similarity:**  
  Assets within the same cluster tend to have similar return and volatility patterns, which often indicates exposure to similar market risks or sectors.
  
- **Diversification Opportunities:**  
  Clustering helps identify assets from different clusters to maximize diversification. Selecting assets from different clusters reduces the correlation within a portfolio, which can lower overall risk.

- **Sector Identification:**  
  Clusters may represent underlying market themes or sectors (e.g., technology, healthcare, energy). This can provide insights into the dominant drivers of asset performance.

- **Risk Management:**  
  Understanding clusters can help in managing sector-specific or cluster-specific risks. For example, if one cluster shows high volatility, you can limit its weight in the portfolio.

- **Behavioral Patterns:**  
  Clustering reveals behavioral patterns in asset performance under different market conditions. This helps identify which groups of assets perform better in bullish or bearish markets.

- **Tactical Asset Allocation:**  
  By observing cluster trends over time, investors can make tactical allocation decisions, such as increasing exposure to clusters with higher returns or reducing exposure to clusters with increasing risk.

- **Portfolio Monitoring:**  
  Over time, clusters can change as assets’ return-risk profiles evolve. Monitoring these shifts can provide early warning signals of changing market dynamics.

- **Tailored Portfolios:**  
  If you have a specific investment objective (e.g., growth vs. income), clustering can guide you to focus on clusters that align with your strategy.
  
- **Efficient Capital Allocation:**  
  Clusters with lower risk and higher returns can guide efficient capital allocation, ensuring resources are invested in the most promising assets or sectors.

""")


if __name__ == "__main__":
    main()
