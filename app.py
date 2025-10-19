import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hi-Pay - Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    .app-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 2px;
    }
    .app-subtitle {
        font-size: 1.2rem;
        color: #f0f0f0;
        margin-top: 0.5rem;
    }
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .dashboard-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 3rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .fraud-detected {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .no-fraud {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open('model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    return model, scaler

model, scaler = load_model()

# Initialize session state for tracking predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'total_checked' not in st.session_state:
    st.session_state.total_checked = 0
if 'fraud_detected' not in st.session_state:
    st.session_state.fraud_detected = 0
if 'total_amount_checked' not in st.session_state:
    st.session_state.total_amount_checked = 0

# Function to preprocess input data
def preprocess_data(data):
    feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFlaggedFraud']
    data['type'] = data['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})
    data_scaled = scaler.transform(data[feature_names])
    return data_scaled

# Create visualizations
def create_fraud_distribution_chart():
    if len(st.session_state.predictions) > 0:
        df = pd.DataFrame(st.session_state.predictions)
        fraud_counts = df['prediction'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraudulent'],
            values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
            hole=0.4,
            marker_colors=['#51cf66', '#ff6b6b'],
            textfont_size=16
        )])
        fig.update_layout(
            title="Transaction Status Distribution",
            showlegend=True,
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    return None

def create_amount_chart():
    if len(st.session_state.predictions) > 0:
        df = pd.DataFrame(st.session_state.predictions)
        df['status'] = df['prediction'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        fig = px.bar(df, x=df.index, y='amount', color='status',
                     color_discrete_map={'Legitimate': '#51cf66', 'Fraudulent': '#ff6b6b'},
                     title='Transaction Amounts by Status')
        fig.update_layout(
            xaxis_title="Transaction #",
            yaxis_title="Amount",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    return None

def create_transaction_type_chart():
    if len(st.session_state.predictions) > 0:
        df = pd.DataFrame(st.session_state.predictions)
        type_counts = df['type'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=type_counts.index,
            y=type_counts.values,
            marker_color='#667eea',
            text=type_counts.values,
            textposition='auto'
        )])
        fig.update_layout(
            title="Transactions by Type",
            xaxis_title="Transaction Type",
            yaxis_title="Count",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    return None

# Main App
def main():
    # Header with Hi-Pay branding
    st.markdown("""
        <div class="title-container">
            <h1 class="app-title">💳 Hi-Pay</h1>
            <p class="app-subtitle">Advanced AI-Powered Fraud Detection System</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 🛡️ About Hi-Pay")
        st.markdown("""
        Hi-Pay uses advanced Machine Learning to detect fraudulent transactions in real-time.
        
        **Model Performance:**
        - Accuracy: 98.45%
        - Algorithm: Random Forest
        - Features: 6 key indicators
        
        **Fraud Indicators:**
        - 🚨 Large transaction amounts
        - 🚨 CASH_OUT & TRANSFER types
        - 🚨 Zero balance origins
        - 🚨 System flagged transactions
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        st.metric("Total Checked", st.session_state.total_checked)
        st.metric("Fraud Detected", st.session_state.fraud_detected)
        if st.session_state.total_checked > 0:
            fraud_rate = (st.session_state.fraud_detected / st.session_state.total_checked) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("### 📝 Transaction Details")
        
        # Form inputs in a cleaner layout
        col_a, col_b = st.columns(2)
        
        with col_a:
            step = st.number_input("⏱️ Time Step (hours)", min_value=1, value=1, 
                                  help="Time elapsed in hours since simulation start")
            amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=0.0, 
                                    format="%.2f", help="Amount to be transferred")
            oldbalanceOrg = st.number_input("🏦 Origin Balance (Before)", min_value=0.0, 
                                           value=0.0, format="%.2f", 
                                           help="Balance of origin account before transaction")
        
        with col_b:
            type_val = st.selectbox("🔄 Transaction Type", 
                                   ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'],
                                   help="Type of financial transaction")
            oldbalanceDest = st.number_input("🏪 Destination Balance (Before)", min_value=0.0, 
                                            value=0.0, format="%.2f",
                                            help="Balance of destination account before transaction")
            isFlaggedFraud = st.checkbox("⚠️ System Flagged", 
                                        help="Check if already flagged by fraud detection system")

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 Analyze Transaction"):
            # Create a DataFrame with user input
            user_data = pd.DataFrame({
                'step': [step],
                'type': [type_val],
                'amount': [amount],
                'oldbalanceOrg': [oldbalanceOrg],
                'oldbalanceDest': [oldbalanceDest],
                'isFlaggedFraud': [isFlaggedFraud]
            })

            # Preprocess the user input
            user_data_scaled = preprocess_data(user_data)

            # Make a prediction
            prediction = model.predict(user_data_scaled)
            probability = model.predict_proba(user_data_scaled)

            # Update session state
            st.session_state.total_checked += 1
            st.session_state.total_amount_checked += amount
            if prediction[0] == 1:
                st.session_state.fraud_detected += 1
            
            st.session_state.predictions.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': type_val,
                'amount': amount,
                'prediction': prediction[0]
            })

            # Display the result with animation
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.markdown(f"""
                    <div class="prediction-box fraud-detected">
                        🚨 FRAUD DETECTED 🚨<br>
                        <span style="font-size: 1rem; opacity: 0.9;">
                        Confidence: {probability[0][1]*100:.1f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ This transaction shows high-risk fraud patterns. Immediate review recommended!")
            else:
                st.markdown(f"""
                    <div class="prediction-box no-fraud">
                        ✅ LEGITIMATE TRANSACTION<br>
                        <span style="font-size: 1rem; opacity: 0.9;">
                        Confidence: {probability[0][0]*100:.1f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                st.success("✓ This transaction appears to be legitimate. Safe to proceed.")

    with col2:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("### 📈 Quick Stats")
        
        # Display metrics
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{st.session_state.total_checked}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                <div class="metric-label">Fraud Detected</div>
                <div class="metric-value">{st.session_state.fraud_detected}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);">
                <div class="metric-label">Total Amount Checked</div>
                <div class="metric-value">${st.session_state.total_amount_checked:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Dashboard section
    if len(st.session_state.predictions) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown("## 📊 Analytics Dashboard")
        
        # Create three columns for charts
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
            fig1 = create_fraud_distribution_chart()
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            fig2 = create_transaction_type_chart()
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        with chart_col3:
            fig3 = create_amount_chart()
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
        
        # Recent transactions table
        st.markdown("### 📋 Recent Transactions")
        df_display = pd.DataFrame(st.session_state.predictions[-10:])
        df_display['Status'] = df_display['prediction'].map({0: '✅ Legitimate', 1: '🚨 Fraudulent'})
        df_display = df_display[['timestamp', 'type', 'amount', 'Status']]
        df_display.columns = ['Timestamp', 'Type', 'Amount', 'Status']
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
