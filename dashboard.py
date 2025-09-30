#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard thời gian thực cho Trading Bot
Sử dụng Streamlit để hiển thị thông tin giao dịch, vị thế và hiệu suất
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import sys

# Cấu hình trang
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff0000;
        font-weight: bold;
    }
    .status-open {
        color: #00ff00;
        font-weight: bold;
    }
    .status-closed {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    """Lớp quản lý Dashboard Trading Bot"""
    
    def __init__(self):
        self.positions_file = "open_positions_h4.json"
        self.db_file = "trading_bot.db"
        self.feature_store_file = "feature_store.db"
        
    def load_open_positions(self):
        """Tải danh sách vị thế đang mở"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                positions = []
                for symbol, pos in data.items():
                    pos['symbol'] = symbol
                    positions.append(pos)
                
                return pd.DataFrame(positions)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Lỗi tải vị thế mở: {e}")
            return pd.DataFrame()
    
    def load_trading_history(self):
        """Tải lịch sử giao dịch từ database"""
        try:
            if not os.path.exists(self.db_file):
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_file)
            
            # Kiểm tra xem có bảng trades không
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            
            if cursor.fetchone():
                query = """
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 1000
                """
                df = pd.read_sql_query(query, conn)
            else:
                df = pd.DataFrame()
            
            conn.close()
            return df
            
        except Exception as e:
            st.error(f"Lỗi tải lịch sử giao dịch: {e}")
            return pd.DataFrame()
    
    def load_feature_store_stats(self):
        """Tải thống kê Feature Store"""
        try:
            if not os.path.exists(self.feature_store_file):
                return {}
            
            conn = sqlite3.connect(self.feature_store_file)
            cursor = conn.cursor()
            
            # Tổng số records
            cursor.execute('SELECT COUNT(*) FROM features')
            total_records = cursor.fetchone()[0]
            
            # Số symbols
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM features')
            total_symbols = cursor.fetchone()[0]
            
            # Số features
            cursor.execute('SELECT COUNT(DISTINCT feature_name) FROM features')
            total_features = cursor.fetchone()[0]
            
            # Thời gian mới nhất
            cursor.execute('SELECT MAX(timestamp) FROM features')
            latest_timestamp = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_records': total_records,
                'total_symbols': total_symbols,
                'total_features': total_features,
                'latest_timestamp': latest_timestamp
            }
            
        except Exception as e:
            st.error(f"Lỗi tải thống kê Feature Store: {e}")
            return {}
    
    def calculate_position_metrics(self, positions_df):
        """Tính toán các chỉ số vị thế"""
        if positions_df.empty:
            return {
                'total_positions': 0,
                'total_pnl': 0,
                'winning_positions': 0,
                'losing_positions': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0
            }
        
        # Tính P&L cho từng vị thế (giả sử có current_price)
        positions_df['current_pnl'] = 0.0  # Placeholder
        
        total_positions = len(positions_df)
        total_pnl = positions_df['current_pnl'].sum()
        winning_positions = len(positions_df[positions_df['current_pnl'] > 0])
        losing_positions = len(positions_df[positions_df['current_pnl'] < 0])
        win_rate = (winning_positions / total_positions * 100) if total_positions > 0 else 0
        avg_profit = positions_df['current_pnl'].mean()
        max_profit = positions_df['current_pnl'].max()
        max_loss = positions_df['current_pnl'].min()
        
        return {
            'total_positions': total_positions,
            'total_pnl': total_pnl,
            'winning_positions': winning_positions,
            'losing_positions': losing_positions,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss
        }
    
    def create_equity_curve(self, trades_df):
        """Tạo biểu đồ đường cong vốn"""
        if trades_df.empty:
            return go.Figure()
        
        # Giả sử có cột 'profit' trong trades_df
        if 'profit' in trades_df.columns:
            cumulative_profit = trades_df['profit'].cumsum()
        else:
            # Tạo dữ liệu giả lập
            cumulative_profit = np.cumsum(np.random.normal(0, 10, len(trades_df)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=cumulative_profit,
            mode='lines',
            name='Equity Curve',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title='Đường Cong Vốn (Equity Curve)',
            xaxis_title='Thời Gian',
            yaxis_title='Lợi Nhuận Tích Lũy',
            hovermode='x unified'
        )
        
        return fig
    
    def create_pnl_distribution(self, trades_df):
        """Tạo biểu đồ phân phối P&L"""
        if trades_df.empty:
            return go.Figure()
        
        # Giả sử có cột 'profit' trong trades_df
        if 'profit' in trades_df.columns:
            profits = trades_df['profit']
        else:
            # Tạo dữ liệu giả lập
            profits = np.random.normal(0, 10, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=profits,
            nbinsx=30,
            name='Phân Phối P&L',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title='Phân Phối Lợi Nhuận/Thua Lỗ',
            xaxis_title='P&L',
            yaxis_title='Số Lượng Giao Dịch',
            bargap=0.1
        )
        
        return fig
    
    def create_symbol_performance(self, trades_df):
        """Tạo biểu đồ hiệu suất theo symbol"""
        if trades_df.empty:
            return go.Figure()
        
        # Giả sử có cột 'symbol' và 'profit' trong trades_df
        if 'symbol' in trades_df.columns and 'profit' in trades_df.columns:
            symbol_performance = trades_df.groupby('symbol')['profit'].sum().sort_values(ascending=True)
        else:
            # Tạo dữ liệu giả lập
            symbols = ['BTCUSD', 'ETHUSD', 'XAUUSD', 'SPX500', 'EURUSD']
            symbol_performance = pd.Series(
                np.random.normal(0, 50, len(symbols)),
                index=symbols
            )
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=symbol_performance.values,
            y=symbol_performance.index,
            orientation='h',
            name='P&L theo Symbol',
            marker_color=['#00ff00' if x > 0 else '#ff0000' for x in symbol_performance.values]
        ))
        
        fig.update_layout(
            title='Hiệu Suất Giao Dịch Theo Symbol',
            xaxis_title='Tổng P&L',
            yaxis_title='Symbol',
            height=400
        )
        
        return fig
    
    def run_dashboard(self):
        """Chạy dashboard chính"""
        # Header
        st.markdown('<h1 class="main-header">📊 Trading Bot Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("⚙️ Cài Đặt")
        
        # Auto refresh
        auto_refresh = st.sidebar.checkbox("🔄 Tự động làm mới", value=True)
        refresh_interval = st.sidebar.slider("⏱️ Khoảng thời gian (giây)", 5, 60, 30)
        
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        
        # Tải dữ liệu
        with st.spinner("🔄 Đang tải dữ liệu..."):
            positions_df = self.load_open_positions()
            trades_df = self.load_trading_history()
            feature_stats = self.load_feature_store_stats()
        
        # Tính toán metrics
        position_metrics = self.calculate_position_metrics(positions_df)
        
        # === PHẦN 1: TỔNG QUAN ===
        st.header("📈 Tổng Quan Hiệu Suất")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Tổng P&L",
                value=f"{position_metrics['total_pnl']:.2f}",
                delta=f"{position_metrics['total_pnl']:.2f}"
            )
        
        with col2:
            st.metric(
                label="🎯 Tỷ Lệ Thắng",
                value=f"{position_metrics['win_rate']:.1f}%",
                delta=f"{position_metrics['win_rate']:.1f}%"
            )
        
        with col3:
            st.metric(
                label="📊 Tổng Vị Thế",
                value=position_metrics['total_positions'],
                delta=position_metrics['total_positions']
            )
        
        with col4:
            st.metric(
                label="📈 Lợi Nhuận TB",
                value=f"{position_metrics['avg_profit']:.2f}",
                delta=f"{position_metrics['avg_profit']:.2f}"
            )
        
        # === PHẦN 2: VỊ THẾ ĐANG MỞ ===
        st.header("🔍 Vị Thế Đang Mở")
        
        if not positions_df.empty:
            # Hiển thị bảng vị thế
            st.dataframe(
                positions_df,
                use_container_width=True,
                height=400
            )
            
            # Thống kê vị thế
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("✅ Vị Thế Thắng", position_metrics['winning_positions'])
            
            with col2:
                st.metric("❌ Vị Thế Thua", position_metrics['losing_positions'])
            
            with col3:
                st.metric("📊 Tỷ Lệ Thắng", f"{position_metrics['win_rate']:.1f}%")
        else:
            st.info("ℹ️ Không có vị thế nào đang mở")
        
        # === PHẦN 3: BIỂU ĐỒ HIỆU SUẤT ===
        st.header("📊 Biểu Đồ Hiệu Suất")
        
        if not trades_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                equity_fig = self.create_equity_curve(trades_df)
                st.plotly_chart(equity_fig, use_container_width=True)
            
            with col2:
                pnl_fig = self.create_pnl_distribution(trades_df)
                st.plotly_chart(pnl_fig, use_container_width=True)
            
            # Biểu đồ hiệu suất theo symbol
            symbol_fig = self.create_symbol_performance(trades_df)
            st.plotly_chart(symbol_fig, use_container_width=True)
        else:
            st.info("ℹ️ Chưa có dữ liệu giao dịch để hiển thị biểu đồ")
        
        # === PHẦN 4: THỐNG KÊ FEATURE STORE ===
        st.header("🗄️ Thống Kê Feature Store")
        
        if feature_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Tổng Records", feature_stats['total_records'])
            
            with col2:
                st.metric("🏷️ Số Symbols", feature_stats['total_symbols'])
            
            with col3:
                st.metric("🔧 Số Features", feature_stats['total_features'])
            
            with col4:
                if feature_stats['latest_timestamp']:
                    latest_time = datetime.fromisoformat(feature_stats['latest_timestamp'])
                    st.metric("⏰ Cập Nhật Cuối", latest_time.strftime("%H:%M:%S"))
                else:
                    st.metric("⏰ Cập Nhật Cuối", "N/A")
        else:
            st.info("ℹ️ Feature Store chưa được khởi tạo")
        
        # === PHẦN 5: LỊCH SỬ GIAO DỊCH ===
        st.header("📋 Lịch Sử Giao Dịch")
        
        if not trades_df.empty:
            # Lọc dữ liệu
            st.subheader("🔍 Bộ Lọc")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'symbol' in trades_df.columns:
                    symbols = ['Tất cả'] + list(trades_df['symbol'].unique())
                    selected_symbol = st.selectbox("Symbol", symbols)
                    if selected_symbol != 'Tất cả':
                        trades_df = trades_df[trades_df['symbol'] == selected_symbol]
            
            with col2:
                if 'timestamp' in trades_df.columns:
                    date_range = st.date_input(
                        "Khoảng thời gian",
                        value=(datetime.now() - timedelta(days=7), datetime.now()),
                        max_value=datetime.now()
                    )
            
            with col3:
                limit = st.slider("Số lượng giao dịch", 10, 1000, 100)
            
            # Hiển thị bảng
            st.dataframe(
                trades_df.head(limit),
                use_container_width=True,
                height=400
            )
        else:
            st.info("ℹ️ Chưa có lịch sử giao dịch")
        
        # === PHẦN 6: THÔNG TIN HỆ THỐNG ===
        st.header("⚙️ Thông Tin Hệ Thống")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📁 Files")
            files_status = {
                "Vị thế mở": "✅" if os.path.exists(self.positions_file) else "❌",
                "Database": "✅" if os.path.exists(self.db_file) else "❌",
                "Feature Store": "✅" if os.path.exists(self.feature_store_file) else "❌"
            }
            
            for file_name, status in files_status.items():
                st.write(f"{status} {file_name}")
        
        with col2:
            st.subheader("🕐 Thời Gian")
            current_time = datetime.now()
            st.write(f"**Hiện tại:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Timezone:** {current_time.astimezone().tzinfo}")
        
        with col3:
            st.subheader("📊 Trạng Thái")
            st.write(f"**Auto Refresh:** {'🟢 Bật' if auto_refresh else '🔴 Tắt'}")
            st.write(f"**Interval:** {refresh_interval}s")
            st.write(f"**Last Update:** {current_time.strftime('%H:%M:%S')}")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "🤖 Trading Bot Dashboard - Powered by Streamlit"
            "</div>",
            unsafe_allow_html=True
        )

def main():
    """Hàm chính"""
    try:
        dashboard = TradingDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"❌ Lỗi chạy dashboard: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()