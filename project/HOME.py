import streamlit as st


def main():
    # Set up the home page
    st.set_page_config(page_title="Portfolio Optimization and Analytics", layout="wide")

    # Add a title and subtitle
    st.title("📊 Portfolio Optimization and Analytics")
    st.subheader("Your One-Stop Solution for Portfolio Analysis, Optimization, and Risk Management")

    # Add a hero section
    st.markdown(
        """
        Welcome to the **Portfolio Optimization and Analytics** application! This platform is designed to help 
        you make informed investment decisions, optimize your portfolio, and analyze risks using advanced financial 
        techniques. Whether you're a seasoned investor or a beginner, this app has something for everyone.
        """
    )

    # Add interactive buttons and navigation
    st.markdown("### Explore the Features:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Portfolio Optimization"):
            st.write("Navigate to the Portfolio Optimization section from the sidebar to analyze and optimize portfolios.")
        
    with col2:
        if st.button("PCA and Forecasting"):
            st.write("Head to the Advance analysis tab to uncover hidden patterns and predict future returns or volatility.")

    with col3:
        if st.button("User-Defined Portfolio"):
            st.write("Go to the User-Defined Portfolio section to define and analyze your custom portfolio allocation.")

    # Use collapsible sections to explain features
    with st.expander("📈 About This Application"):
        st.markdown(
            """
            This application leverages advanced financial models and data science techniques to provide insights 
            into portfolio management. Here's what you can do:
            - **Portfolio Optimization**: Use Monte Carlo simulations and mean-variance optimization to find the 
              best portfolio allocation for your investments.
            - **PCA and Forecasting**: Apply Principal Component Analysis (PCA) to reduce dimensionality and forecast 
              future returns or volatility with ARCH/GARCH models.
            - **User-Defined Portfolios**: Input your custom portfolio and calculate weights, risk metrics like VaR/CVaR, 
              and cluster analysis.
            - **Risk Management**: Analyze portfolio risks using efficient frontiers, Sharpe ratios, and clustering techniques.
            """
        )

    with st.expander("🔍 Use Cases for This Application"):
        st.markdown(
            """
            This tool is ideal for:
            - **Individual Investors**: Optimize personal portfolios based on historical data and risk tolerance.
            - **Financial Analysts**: Perform in-depth risk and return analysis using cutting-edge financial models.
            - **Students and Educators**: Learn about portfolio management, PCA, and advanced forecasting techniques.
            - **Portfolio Managers**: Analyze large datasets to make data-driven investment decisions.
            """
        )

    # Add an image or visual representation (optional)
    st.image(
        "https://miro.medium.com/v2/resize:fit:825/1*TtTQVg3OKWOjkwPxoKG2Fg.jpeg",
        caption="Visualize and Optimize Your Portfolio",
        use_container_width=True,
        
    )

    
    # Disclaimer section
    st.markdown(
        """
        ---
        **Disclaimer:** This application is for informational and educational purposes only. It does not constitute 
        financial, investment, or legal advice. Users are encouraged to consult with a qualified financial advisor 
        or conduct their own research before making any investment decisions. The app creators assume no responsibility 
        for any losses or liabilities incurred from the use of this application.
        """
    )

    # Call to action
    st.markdown(
        """
        ---
        Ready to dive in? Use the sidebar to navigate through the app and start optimizing your portfolio today! 🚀
        """
    )

if __name__ == "__main__":
    main()
