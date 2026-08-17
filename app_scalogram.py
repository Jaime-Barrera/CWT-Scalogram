import streamlit as st
import numpy as np
import pywt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import data_extractor as de


st.set_page_config(
    page_title="Scalogram Generator",
    page_icon="",
    layout="wide"
)




def init_session_state():
    defaults = {
        "ticker": "AAPL",
        "start_date": datetime.today() - timedelta(days=365*2),
        "end_date": datetime.today(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=3600)
def fetch_data(ticker: str, start: str, end: str):
    dates, prices = de.get_data(ticker, start=start, end=end)
    return dates, prices


###################################
#          DENOISING              #  
###################################

def sure_shrink_threshold(data: np.ndarray) -> float:
    """
    SUREShrink: Stein's Unbiased Risk Estimate for optimal threshold selection.
    
    This method minimizes the statistical risk (MSE) when denoising.
    Falls back to Universal Threshold if data is too sparse (many near-zero values).
    
    Reference: Donoho & Johnstone (1995) "Adapting to Unknown Smoothness via Wavelet Shrinkage"
    """
    n = len(data)
    if n == 0:
        return 0.0
    
    sigma = 1.4826 * np.median(np.abs(data - np.median(data)))
    
    if sigma < 1e-10:
        return 0.0
    
    data_normalized = data / sigma
    
    sorted_squared = np.sort(data_normalized ** 2)
    
    cumsum = np.cumsum(sorted_squared)
    risks = (n - 2 * np.arange(1, n + 1) + cumsum + (n - np.arange(1, n + 1)) * sorted_squared) / n
    
    min_risk_idx = np.argmin(risks)
    sure_threshold = np.sqrt(sorted_squared[min_risk_idx]) * sigma
    
    universal_threshold = sigma * np.sqrt(2 * np.log(n))
    
    sparsity_check = np.sum(sorted_squared - 1) / n
    if sparsity_check <= (np.log2(n) ** 1.5) / np.sqrt(n):
        return universal_threshold
    
    return min(sure_threshold, universal_threshold)


def denoise_returns(log_returns: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """
    Apply soft thresholding to log-returns.
    If threshold is None, uses SUREShrink adaptive threshold.
    """
    if threshold is None:
        threshold = sure_shrink_threshold(log_returns)
    return pywt.threshold(log_returns, threshold, mode='soft')


###################################
#       COMPUTE SCALOGRAM         #  
###################################

@st.cache_data(ttl=60)
def compute_scalogram(prices, max_scale: int = 128, denoise: bool = True, manual_threshold: float | None = None):
    log_returns = np.diff(np.log(prices))
    log_returns = np.nan_to_num(log_returns)
    
    auto_threshold = sure_shrink_threshold(log_returns)
    
    if denoise:
        threshold = manual_threshold if manual_threshold is not None else auto_threshold
        log_returns = denoise_returns(log_returns, threshold)
    
    scales = np.arange(1, max_scale)
    coeffs, freqs = pywt.cwt(log_returns, scales, 'cmor1.5-1.0')
    power = (np.abs(coeffs)) ** 2
    
    return power, scales, log_returns, auto_threshold



###################################
#       SCALOGRAM DESIGN          #  
###################################

def create_scalogram_figure(power, dates, scales, ticker: str):
    fig = go.Figure()
    
    date_labels = [d.strftime('%Y-%m-%d') for d in dates[1:]]
    
    fig.add_trace(go.Heatmap(
        z=power,
        x=date_labels,
        y=scales,
        colorscale='Jet',
        colorbar=dict(title="Energy (Volatility)"),
        hovertemplate="Date: %{x}<br>Scale: %{y}<br>Power: %{z:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Volatility Scalogram: {ticker}",
            font=dict(size=20, color="#333333")
        ),
        xaxis=dict(
            title="Date",
            showgrid=False,
            tickangle=45,
            tickfont=dict(color="#333333"),
            title_font=dict(color="#333333")
        ),
        yaxis=dict(
            title="Scale (Inverse Frequency)<br>Lower=Fast, Upper=Slow",
            showgrid=False,
            tickfont=dict(color="#333333"),
            title_font=dict(color="#333333")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=20, t=60, b=80),
        height=600
    )
    
    return fig


###################################
#           SIDEBAR               #  
###################################

def setup_sidebar():
    with st.sidebar:
        st.markdown("## Settings")
        st.markdown("---")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            placeholder="e.g., AAPL, NVDA, BTC-USD",
            help="Enter any valid Yahoo Finance ticker symbol"
        ).upper().strip()
        
        if not ticker:
            ticker = "AAPL"
        
        st.markdown("### Date Range")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.today() - timedelta(days=365*2),
                max_value=datetime.today() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.today(),
                max_value=datetime.today(),
                help="Choose a date at least 30 days after the start date"
            )
        
        max_scale = st.slider(
            "Max Scale",
            min_value=32,
            max_value=256,
            value=128,
            step=16,
            help="Higher scales capture slower (longer-term) volatility patterns"
        )
        
        st.markdown("### Denoising")
        denoise_enabled = st.checkbox("Enable Denoising", value=True, help="Reduce noise for cleaner patterns")
        
        manual_threshold = None
        if denoise_enabled:
            advanced_mode = st.checkbox("Manual threshold", value=False)
            if advanced_mode:
                manual_threshold = st.slider(
                    "Threshold",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Lower = more aggressive filtering, Higher = preserve more detail"
                )
        
        st.markdown("---")
        generate_btn = st.button(
            "Generate Scalogram",
            type="primary",
            use_container_width=True
        )
        
    return ticker, str(start_date), str(end_date), max_scale, denoise_enabled, manual_threshold, generate_btn


###################################
#           HEADER                #  
###################################

def render_header():
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 2px solid #333; margin-bottom: 1.5rem;">
        <h1 style="margin: 0; color: #000000;">Wavelet Scalogram</h1>
        <p style="margin: 0; color: #888;">Interactive volatility analysis using Continuous Wavelet Transform</p>
    </div>
    """, unsafe_allow_html=True)


###################################
#           MAIN                  #  
###################################

def main():
    init_session_state()
    
    ticker, start_date, end_date, max_scale, denoise_enabled, manual_threshold, generate = setup_sidebar()
    
    render_header()
    
    if generate or "last_params" not in st.session_state:
        st.session_state.last_params = (ticker, start_date, end_date, max_scale, denoise_enabled, manual_threshold)
    
    params = st.session_state.get("last_params", (ticker, start_date, end_date, max_scale, denoise_enabled, manual_threshold))
    ticker, start_date, end_date, max_scale, denoise_enabled, manual_threshold = params
    
    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            dates, prices = fetch_data(ticker, start_date, end_date)
        
        if len(prices) < 30:
            st.error("Not enough data points. Please expand the date range.")
            st.stop()
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.markdown(f"""
            <div style="padding: 0.5rem 0;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Ticker</p>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">{ticker}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_info2:
            st.markdown(f"""
            <div style="padding: 0.5rem 0;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Data Points</p>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">{len(prices):,}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_info3:
            st.markdown(f"""
            <div style="padding: 0.5rem 0;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Date Range</p>
                <p style="margin: 0; font-size: 1.3rem; font-weight: 600;">{start_date} to {end_date}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Computing scalogram..."):
            power, scales, log_returns, auto_threshold = compute_scalogram(
                tuple(prices), max_scale, denoise_enabled, manual_threshold
            )
        
        with col_info4:
            threshold_used = manual_threshold if manual_threshold else auto_threshold
            denoise_status = f"{threshold_used:.4f}" if denoise_enabled else "Off"
            st.markdown(f"""
            <div style="padding: 0.5rem 0;">
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Denoising</p>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">{denoise_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        fig = create_scalogram_figure(power, dates, scales, ticker)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        with st.expander("Interpretation Guide"):
            st.markdown("""
            **How to read the Scalogram:**
            
            - **X-Axis (Date):** Time progression of the financial data
            - **Y-Axis (Scale):** Frequency of volatility patterns
                - Lower scales = High-frequency (short-term) volatility
                - Higher scales = Low-frequency (long-term) volatility trends
            - **Color (Energy):** Intensity of volatility at each time-scale point
                - Red/Yellow = High volatility energy
                - Blue/Cyan = Low volatility energy
            
            **Key patterns to look for:**
            - Vertical bands indicate volatility spikes across all frequencies
            - Horizontal bands suggest persistent volatility at specific time scales
            - Cone-shaped patterns may indicate volatility cascading from short to long term
            """)

        with st.expander("Theoretical Background"):
            st.markdown("""
            **For the denoising process:**
            - The method used is SUREShrink (Stein's Unbiased Risk Estimate)
            - It works by estimating the noise level in the wavelet coefficients and thresholding them
            - The threshold is chosen to minimize the mean squared error of the denoised signal
            - If the data is too complex, the denoising will fallback to the universal threshold estimate
            
            **For the scalogram process:**
            - The method used is the Continuous Wavelet Transform (CWT)
            - It works by convolving the signal with a complex wavelet
            - The result is a time-frequency representation of the signal
            - The scalogram represents the local energy (volatility) of the signal at different time scales
            - For higher frequencies it offers better time resolution
            - For lower frequencies it offers better frequency resolution, allowing to detect long-term trends (volatility spikes)
            
            
            """)
            
    except ValueError as ve:
        st.error(f"Data error: {str(ve)}")
        st.info("Check that the asset symbol is valid and that the date range contains trading days.")
    except Exception as e:
        st.error(f"Unexpected system error: {str(e)}")
        st.info("Please check the console for more details or try different parameters.")

if __name__ == "__main__":
    main()
