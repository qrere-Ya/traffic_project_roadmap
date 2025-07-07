"""
交通數據視覺化模組
======================

功能：
1. 7天時間序列分析圖表
2. 尖峰離峰對比視覺化
3. AI模型評分圖表
4. 數據品質熱力圖
5. 互動式交通流量儀表板
6. 車種行為分析圖

基於：80,640筆AI訓練數據 + 7天完整週期
作者: 交通預測專案團隊
日期: 2025-07-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體和樣式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class TrafficVisualizer:
    """交通數據視覺化器"""
    
    def __init__(self, base_folder: str = "data"):
        self.base_folder = Path(base_folder)
        self.cleaned_folder = self.base_folder / "cleaned"
        self.output_folder = Path("outputs/figures")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 數據容器
        self.datasets = {}
        self.ai_analysis = {}
        
        print("🎨 交通數據視覺化模組初始化...")
        print(f"   📁 數據目錄: {self.cleaned_folder}")
        print(f"   📊 輸出目錄: {self.output_folder}")
        
        # 載入數據和分析結果
        self._load_visualization_data()
    
    def _load_visualization_data(self):
        """載入視覺化所需數據"""
        print("📊 載入視覺化數據...")
        
        try:
            # 使用簡化版分析器載入數據
            from flow_analyzer import SimplifiedTrafficAnalyzer
            
            analyzer = SimplifiedTrafficAnalyzer()
            if analyzer.load_data(merge_dates=True):
                self.datasets = analyzer.datasets
                
                # 執行分析以獲取AI評估結果
                analyzer.analyze_data_characteristics()
                analyzer.evaluate_ai_model_suitability()
                self.ai_analysis = analyzer.analysis_results
                
                # 統計載入情況
                total_records = sum(len(df) for df in self.datasets.values())
                print(f"   ✅ 成功載入 {len(self.datasets)} 個數據集")
                print(f"   📈 總記錄數: {total_records:,} 筆")
                
                # 檢查AI分析結果
                if 'ai_evaluation' in self.ai_analysis:
                    recommendations = self.ai_analysis['ai_evaluation']['recommendations']
                    print(f"   🤖 AI模型推薦: {len(recommendations)} 個模型")
                
                return True
            else:
                print("   ❌ 數據載入失敗")
                return False
                
        except Exception as e:
            print(f"   ❌ 視覺化數據載入錯誤: {e}")
            return False
    
    def plot_time_series_analysis(self, save_html: bool = True):
        """7天時間序列分析圖表"""
        print("📈 生成7天時間序列分析圖...")
        
        if 'target_peak' not in self.datasets or 'target_offpeak' not in self.datasets:
            print("   ❌ 缺少目標路段數據")
            return None
        
        # 準備數據
        peak_df = self.datasets['target_peak'].copy()
        offpeak_df = self.datasets['target_offpeak'].copy()
        
        # 確保時間欄位正確
        for df in [peak_df, offpeak_df]:
            if 'update_time' in df.columns:
                df['update_time'] = pd.to_datetime(df['update_time'])
            if 'source_date' in df.columns:
                df['date'] = pd.to_datetime(df['source_date'])
        
        # 創建互動式時間序列圖
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '📊 每日平均速度趨勢', '🚗 每日平均流量趨勢',
                '⏰ 24小時速度模式', '🕐 24小時流量模式',
                '📅 尖峰離峰對比', '🎯 數據品質分布'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 每日平均速度趨勢
        if 'source_date' in peak_df.columns:
            daily_peak_speed = peak_df.groupby('source_date')['speed'].mean()
            daily_offpeak_speed = offpeak_df.groupby('source_date')['speed'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_peak_speed.index,
                    y=daily_peak_speed.values,
                    name='尖峰時段平均速度',
                    line=dict(color='red', width=3),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_offpeak_speed.index,
                    y=daily_offpeak_speed.values,
                    name='離峰時段平均速度',
                    line=dict(color='blue', width=3),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # 2. 每日平均流量趨勢
        if 'source_date' in peak_df.columns:
            daily_peak_volume = peak_df.groupby('source_date')['volume_total'].mean()
            daily_offpeak_volume = offpeak_df.groupby('source_date')['volume_total'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_peak_volume.index,
                    y=daily_peak_volume.values,
                    name='尖峰時段平均流量',
                    line=dict(color='orange', width=3),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_offpeak_volume.index,
                    y=daily_offpeak_volume.values,
                    name='離峰時段平均流量',
                    line=dict(color='green', width=3),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
        
        # 3. 24小時速度模式（如果有時間數據）
        if 'update_time' in peak_df.columns:
            peak_df['hour'] = peak_df['update_time'].dt.hour
            offpeak_df['hour'] = offpeak_df['update_time'].dt.hour
            
            hourly_peak_speed = peak_df.groupby('hour')['speed'].mean()
            hourly_offpeak_speed = offpeak_df.groupby('hour')['speed'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_peak_speed.index,
                    y=hourly_peak_speed.values,
                    name='尖峰小時速度',
                    line=dict(color='crimson', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_offpeak_speed.index,
                    y=hourly_offpeak_speed.values,
                    name='離峰小時速度',
                    line=dict(color='steelblue', width=2)
                ),
                row=2, col=1
            )
        
        # 4. 24小時流量模式
        if 'hour' in peak_df.columns:
            hourly_peak_volume = peak_df.groupby('hour')['volume_total'].mean()
            hourly_offpeak_volume = offpeak_df.groupby('hour')['volume_total'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_peak_volume.index,
                    y=hourly_peak_volume.values,
                    name='尖峰小時流量',
                    line=dict(color='darkorange', width=2)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_offpeak_volume.index,
                    y=hourly_offpeak_volume.values,
                    name='離峰小時流量',
                    line=dict(color='darkgreen', width=2)
                ),
                row=2, col=2
            )
        
        # 5. 尖峰離峰記錄數對比
        peak_count = len(peak_df)
        offpeak_count = len(offpeak_df)
        
        fig.add_trace(
            go.Bar(
                x=['尖峰時段', '離峰時段'],
                y=[peak_count, offpeak_count],
                name='記錄數量',
                marker_color=['red', 'blue'],
                text=[f'{peak_count:,}筆', f'{offpeak_count:,}筆'],
                textposition='auto'
            ),
            row=3, col=1
        )
        
        # 6. 數據品質分布（基於AI分析結果）
        if 'data_characteristics' in self.ai_analysis:
            quality_metrics = self.ai_analysis['data_characteristics']['quality_metrics']
            
            metrics_names = ['數據量評分', '時間評分', '平衡性評分', '整體品質']
            metrics_values = [
                quality_metrics.get('volume_score', 0),
                quality_metrics.get('time_score', 0),
                quality_metrics.get('balance_score', 0),
                quality_metrics.get('overall_quality', 0)
            ]
            
            colors = ['lightcoral' if v < 70 else 'lightgreen' if v < 85 else 'gold' for v in metrics_values]
            
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    name='品質評分',
                    marker_color=colors,
                    text=[f'{v:.1f}' for v in metrics_values],
                    textposition='auto'
                ),
                row=3, col=2
            )

        # 🔧 正確獲取 recommendations 數據
        recommendations = []
        if 'ai_evaluation' in self.ai_analysis:
            ai_eval = self.ai_analysis['ai_evaluation']
            if 'recommendations' in ai_eval and ai_eval['recommendations']:
                recommendations = ai_eval['recommendations']

        # 在主圖表下方添加獨立的雷達圖
        if recommendations and len(recommendations) > 0:
            # 創建獨立的雷達圖
            radar_fig = go.Figure()
            
            top_models = recommendations[:3]
            categories = ['預測精度', '訓練速度', '解釋性', '資源需求', '適用性']
            colors = ['gold', 'silver', '#CD7F32']
            
            for i, rec in enumerate(top_models):
                model_name = rec['model']
                score = rec['score']
                
                # 根據模型特性設定雷達圖數值
                if 'lstm' in model_name.lower():
                    values = [score, 60, 40, 30, 90]  # LSTM特性
                elif 'xgboost' in model_name.lower():
                    values = [score, 80, 70, 50, 85]  # XGBoost特性
                elif 'random_forest' in model_name.lower():
                    values = [score, 85, 80, 60, 80]  # 隨機森林特性
                else:
                    values = [score, 70, 60, 50, 70]  # 預設值
                
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f"{rec['priority']} {model_name}",
                        line_color=colors[i]
                    )
                )
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="🎯 推薦模型特性雷達圖",
                template='plotly_white'
            )
            
            # 保存獨立雷達圖
            radar_output_path = self.output_folder / "ai_model_radar_chart.html"
            radar_fig.write_html(str(radar_output_path))
            print(f"   ✅ 模型雷達圖已保存: {radar_output_path}")
        
        fig.update_layout(
            title={
                'text': '🚀 國道1號圓山-三重路段：7天時間序列完整分析',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # 保存圖表
        if save_html:
            output_path = self.output_folder / "time_series_analysis.html"
            fig.write_html(str(output_path))
            print(f"   ✅ 時間序列圖表已保存: {output_path}")
        
        return fig
    
    def plot_ai_model_recommendations(self, save_html: bool = True):
        """AI模型推薦評分圖表"""
        print("🤖 生成AI模型推薦圖表...")
        
        if 'ai_evaluation' not in self.ai_analysis:
            print("   ❌ 缺少AI評估結果")
            return None
        
        ai_eval = self.ai_analysis['ai_evaluation']
        model_suitability = ai_eval['model_suitability']
        recommendations = ai_eval['recommendations']
        data_readiness = ai_eval['data_readiness']
        
        # 創建模型評分圖表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🎯 AI模型適用性評分排行',
                '📊 推薦模型詳細評分',
                '🚀 數據就緒度評估',
                '💡 模型推薦優勢分析'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # 1. 模型評分排行榜
        model_names = list(model_suitability.keys())
        model_scores = [model_suitability[name]['score'] for name in model_names]
        model_suitable = [model_suitability[name]['suitable'] for name in model_names]
        
        # 根據適用性設定顏色
        colors = ['green' if suitable else 'lightcoral' for suitable in model_suitable]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=model_scores,
                name='模型評分',
                marker_color=colors,
                text=[f'{score:.1f}分' for score in model_scores],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. 推薦模型詳細評分（改用條形圖）
        if recommendations:
            top_models = recommendations[:3]
            model_names = [rec['model'] for rec in top_models]
            model_scores = [rec['score'] for rec in top_models]
            priorities = [rec['priority'] for rec in top_models]
            
            # 設定顏色
            colors = ['gold', 'silver', '#CD7F32']  # 金、銀、銅色
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=model_scores,
                    name='推薦模型評分',
                    marker_color=colors[:len(model_names)],
                    text=[f'{score:.1f}分<br>{priority}' for score, priority in zip(model_scores, priorities)],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. 數據就緒度指標
        lstm_ready = data_readiness.get('lstm_ready', False)
        production_ready = data_readiness.get('production_ready', False)
        records = data_readiness.get('records', 0)
        time_span = data_readiness.get('time_span', 0)
        quality = data_readiness.get('quality', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "數據品質評分"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ),
            row=2, col=1
        )
        
        # 4. 就緒度評估
        readiness_metrics = ['LSTM就緒', '生產就緒', '數據充足', '時間充足']
        readiness_values = [
            100 if lstm_ready else 50,
            100 if production_ready else 50,
            100 if records >= 50000 else (records / 50000 * 100),
            100 if time_span >= 7 else (time_span / 7 * 100)
        ]
        
        readiness_colors = ['green' if v >= 80 else 'orange' if v >= 50 else 'red' for v in readiness_values]
        
        fig.add_trace(
            go.Bar(
                x=readiness_metrics,
                y=readiness_values,
                name='就緒度',
                marker_color=readiness_colors,
                text=[f'{v:.0f}%' for v in readiness_values],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title={
                'text': f'🤖 AI模型智能推薦系統分析結果 (基於{records:,}筆訓練數據)',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # 添加註解
        annotations_text = []
        if recommendations:
            annotations_text.append(f"🥇 首選模型: {recommendations[0]['model']} ({recommendations[0]['score']:.1f}分)")
            if len(recommendations) > 1:
                annotations_text.append(f"🥈 次選模型: {recommendations[1]['model']} ({recommendations[1]['score']:.1f}分)")
        
        annotations_text.append(f"📊 AI訓練數據: {records:,}筆 | 時間跨度: {time_span}天")
        annotations_text.append(f"🎯 LSTM就緒: {'✅ 是' if lstm_ready else '❌ 否'}")
        
        fig.add_annotation(
            text="<br>".join(annotations_text),
            xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            font=dict(size=12, color="darkblue"),
            bgcolor="lightblue", bordercolor="blue", borderwidth=1
        )
        
        # 保存圖表
        if save_html:
            output_path = self.output_folder / "ai_model_recommendations.html"
            fig.write_html(str(output_path))
            print(f"   ✅ AI模型推薦圖表已保存: {output_path}")
        
        return fig
    
    def plot_vehicle_type_analysis(self, save_html: bool = True):
        """車種行為分析圖表"""
        print("🚗 生成車種行為分析圖...")
        
        if 'target_peak' not in self.datasets:
            print("   ❌ 缺少目標路段數據")
            return None
        
        # 合併尖峰離峰數據
        all_data = []
        if 'target_peak' in self.datasets:
            df_peak = self.datasets['target_peak'].copy()
            df_peak['period_type'] = '尖峰時段'
            all_data.append(df_peak)
        
        if 'target_offpeak' in self.datasets:
            df_offpeak = self.datasets['target_offpeak'].copy()
            df_offpeak['period_type'] = '離峰時段'
            all_data.append(df_offpeak)
        
        if not all_data:
            print("   ❌ 無可用數據")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 創建車種分析圖表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🚗 車種流量分布', '⚡ 各車種平均速度',
                '📊 尖峰離峰車種對比', '🎯 車種佔比分析'
            ),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. 車種流量分布
        vehicle_columns = ['volume_small', 'volume_large', 'volume_truck']
        vehicle_names = ['小車', '大車', '卡車']
        vehicle_colors = ['lightblue', 'orange', 'lightcoral']
        
        for i, (col, name, color) in enumerate(zip(vehicle_columns, vehicle_names, vehicle_colors)):
            if col in combined_df.columns:
                avg_volume = combined_df[col].mean()
                fig.add_trace(
                    go.Bar(
                        x=[name],
                        y=[avg_volume],
                        name=name,
                        marker_color=color,
                        text=f'{avg_volume:.1f}',
                        textposition='auto'
                    ),
                    row=1, col=1
                )
        
        # 2. 各車種平均速度箱型圖
        speed_columns = ['speed_small', 'speed_large', 'speed_truck']
        speed_names = ['小車速度', '大車速度', '卡車速度']
        
        for i, (col, name, color) in enumerate(zip(speed_columns, speed_names, vehicle_colors)):
            if col in combined_df.columns and combined_df[col].notna().any():
                fig.add_trace(
                    go.Box(
                        y=combined_df[col].dropna(),
                        name=name,
                        marker_color=color,
                        boxpoints='outliers'
                    ),
                    row=1, col=2
                )
        
        # 3. 尖峰離峰車種對比
        for period in ['尖峰時段', '離峰時段']:
            period_data = combined_df[combined_df['period_type'] == period]
            if not period_data.empty:
                volumes = []
                for col in vehicle_columns:
                    if col in period_data.columns:
                        volumes.append(period_data[col].mean())
                    else:
                        volumes.append(0)
                
                fig.add_trace(
                    go.Bar(
                        x=vehicle_names,
                        y=volumes,
                        name=period,
                        text=[f'{v:.1f}' for v in volumes],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
        
        # 4. 車種佔比分析（基於總流量）
        total_volumes = []
        for col in vehicle_columns:
            if col in combined_df.columns:
                total_volumes.append(combined_df[col].sum())
            else:
                total_volumes.append(0)
        
        if sum(total_volumes) > 0:
            fig.add_trace(
                go.Pie(
                    labels=vehicle_names,
                    values=total_volumes,
                    hole=0.3,
                    marker_colors=vehicle_colors,
                    textinfo='label+percent',
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title={
                'text': '🚗 國道1號圓山-三重路段：車種行為深度分析',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # 保存圖表
        if save_html:
            output_path = self.output_folder / "vehicle_type_analysis.html"
            fig.write_html(str(output_path))
            print(f"   ✅ 車種分析圖表已保存: {output_path}")
        
        return fig
    
    def create_interactive_dashboard(self, save_html: bool = True):
        """創建互動式交通流量儀表板"""
        print("📊 創建互動式交通流量儀表板...")
        
        if not self.datasets:
            print("   ❌ 無可用數據")
            return None
        
        # 計算關鍵指標
        total_records = sum(len(df) for df in self.datasets.values())
        
        # 獲取AI分析結果
        ai_ready = False
        lstm_score = 0
        top_model = "無推薦"
        
        if 'ai_evaluation' in self.ai_analysis:
            data_readiness = self.ai_analysis['ai_evaluation']['data_readiness']
            ai_ready = data_readiness.get('lstm_ready', False)
            
            recommendations = self.ai_analysis['ai_evaluation']['recommendations']
            if recommendations:
                top_model = recommendations[0]['model']
                lstm_score = recommendations[0]['score']
        
        # 創建儀表板
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '📊 總數據量', '🤖 AI就緒度', '🥇 推薦模型',
                '📈 尖峰時段流量', '📉 離峰時段流量', '⚡ 平均速度',
                '🎯 數據品質', '📅 時間跨度', '🚀 系統狀態'
            ),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # 第一行指標
        # 1. 總數據量
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=total_records,
                title={"text": "總記錄數"},
                delta={"reference": 50000, "valueformat": ","},
                number={"valueformat": ","},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # 2. AI就緒度
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=100 if ai_ready else 50,
                title={"text": "AI開發就緒度"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green" if ai_ready else "orange"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "lightgreen"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}
                }
            ),
            row=1, col=2
        )
        
        # 3. 推薦模型評分
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=lstm_score,
                title={"text": f"推薦模型評分<br>({top_model})"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "gold"},
                    'steps': [{'range': [0, 60], 'color': "lightgray"},
                             {'range': [60, 80], 'color': "yellow"},
                             {'range': [80, 100], 'color': "gold"}]
                }
            ),
            row=1, col=3
        )
        
        # 第二行指標
        # 計算實際交通指標
        if 'target_peak' in self.datasets and not self.datasets['target_peak'].empty:
            peak_df = self.datasets['target_peak']
            avg_peak_volume = peak_df['volume_total'].mean() if 'volume_total' in peak_df.columns else 0
            avg_peak_speed = peak_df['speed'].mean() if 'speed' in peak_df.columns else 0
        else:
            avg_peak_volume = 0
            avg_peak_speed = 0
        
        if 'target_offpeak' in self.datasets and not self.datasets['target_offpeak'].empty:
            offpeak_df = self.datasets['target_offpeak']
            avg_offpeak_volume = offpeak_df['volume_total'].mean() if 'volume_total' in offpeak_df.columns else 0
            avg_offpeak_speed = offpeak_df['speed'].mean() if 'speed' in offpeak_df.columns else 0
        else:
            avg_offpeak_volume = 0
            avg_offpeak_speed = 0
        
        overall_avg_speed = (avg_peak_speed + avg_offpeak_speed) / 2 if (avg_peak_speed + avg_offpeak_speed) > 0 else 0
        
        # 4. 尖峰時段流量
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=avg_peak_volume,
                title={"text": "尖峰平均流量<br>(輛/分鐘)"},
                number={"valueformat": ".1f"},
                gauge={'axis': {'range': [None, 50]}, 'bar': {'color': "red"}}
            ),
            row=2, col=1
        )
        
        # 5. 離峰時段流量
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=avg_offpeak_volume,
                title={"text": "離峰平均流量<br>(輛/分鐘)"},
                number={"valueformat": ".1f"},
                gauge={'axis': {'range': [None, 50]}, 'bar': {'color': "blue"}}
            ),
            row=2, col=2
        )
        
        # 6. 平均速度
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=overall_avg_speed,
                title={"text": "整體平均速度<br>(km/h)"},
                number={"valueformat": ".1f"},
                gauge={'axis': {'range': [None, 120]}, 'bar': {'color': "green"}}
            ),
            row=2, col=3
        )
        
        # 第三行指標
        # 7. 數據品質
        quality_score = 0
        if 'data_characteristics' in self.ai_analysis:
            quality_score = self.ai_analysis['data_characteristics']['quality_metrics'].get('overall_quality', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title={"text": "數據品質評分"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"},
                    'steps': [{'range': [0, 60], 'color': "lightgray"},
                             {'range': [60, 80], 'color': "yellow"},
                             {'range': [80, 100], 'color': "lightgreen"}]
                }
            ),
            row=3, col=1
        )
        
        # 8. 時間跨度
        time_span = 0
        if 'data_characteristics' in self.ai_analysis:
            time_span = self.ai_analysis['data_characteristics']['time_coverage'].get('date_span_days', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=time_span,
                title={"text": "數據時間跨度<br>(天數)"},
                delta={"reference": 7},
                number={"suffix": " 天"}
            ),
            row=3, col=2
        )
        
        # 9. 系統狀態
        system_status = 100 if (ai_ready and quality_score >= 70 and time_span >= 7) else 70
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=system_status,
                title={"text": "系統整體狀態"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "gold" if system_status >= 90 else "orange"},
                    'steps': [{'range': [0, 60], 'color': "lightcoral"},
                             {'range': [60, 90], 'color': "lightyellow"},
                             {'range': [90, 100], 'color': "lightgreen"}]
                }
            ),
            row=3, col=3
        )
        
        # 更新布局
        fig.update_layout(
            title={
                'text': f'🚀 智慧交通預測系統即時儀表板<br><sub>國道1號圓山-三重路段 | 數據更新時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</sub>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=1000,
            template='plotly_white',
            font=dict(size=14)
        )
        
        # 添加狀態說明
        status_text = []
        if ai_ready:
            status_text.append("🤖 AI模型開發：✅ 就緒")
        else:
            status_text.append("🤖 AI模型開發：⚠️ 準備中")
        
        status_text.append(f"📊 總數據量：{total_records:,}筆記錄")
        status_text.append(f"🎯 推薦模型：{top_model}")
        status_text.append(f"📈 數據品質：{quality_score:.1f}/100")
        
        fig.add_annotation(
            text="<br>".join(status_text),
            xref="paper", yref="paper",
            x=0.02, y=0.98, showarrow=False,
            font=dict(size=12, color="darkgreen"),
            bgcolor="lightgreen", bordercolor="green", borderwidth=2
        )
        
        # 保存儀表板
        if save_html:
            output_path = self.output_folder / "interactive_dashboard.html"
            fig.write_html(str(output_path))
            print(f"   ✅ 互動式儀表板已保存: {output_path}")
        
        return fig
    
    def plot_data_quality_heatmap(self, save_html: bool = True):
        """數據品質熱力圖"""
        print("🔥 生成數據品質熱力圖...")
        
        # 檢查是否有按日期分組的數據
        date_quality_data = []
        
        # 嘗試從各數據集提取日期信息
        for dataset_name, df in self.datasets.items():
            if df.empty or 'source_date' not in df.columns:
                continue
            
            # 按日期分組計算品質指標
            date_groups = df.groupby('source_date')
            
            for date_str, date_df in date_groups:
                # 計算各項品質指標
                completeness = {}
                numeric_columns = ['speed', 'occupancy', 'volume_total', 'volume_small', 'volume_large', 'volume_truck']
                
                for col in numeric_columns:
                    if col in date_df.columns:
                        total_values = len(date_df)
                        valid_values = date_df[col].notna().sum()
                        completeness[col] = (valid_values / total_values * 100) if total_values > 0 else 0
                    else:
                        completeness[col] = 0
                
                # 計算異常值比例
                anomaly_rate = 0
                if 'speed' in date_df.columns:
                    invalid_speed = ((date_df['speed'] < 0) | (date_df['speed'] > 150)).sum()
                    anomaly_rate = (invalid_speed / len(date_df) * 100) if len(date_df) > 0 else 0
                
                date_quality_data.append({
                    'date': date_str,
                    'dataset': dataset_name,
                    'records': len(date_df),
                    'speed_completeness': completeness.get('speed', 0),
                    'volume_completeness': completeness.get('volume_total', 0),
                    'occupancy_completeness': completeness.get('occupancy', 0),
                    'anomaly_rate': 100 - anomaly_rate,  # 轉換為品質分數
                    'overall_quality': np.mean(list(completeness.values()) + [100 - anomaly_rate])
                })
        
        if not date_quality_data:
            print("   ⚠️ 無法生成品質熱力圖，缺少日期分組數據")
            return None
        
        # 轉換為DataFrame
        quality_df = pd.DataFrame(date_quality_data)
        
        # 創建熱力圖
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📅 各日期數據品質熱力圖', '📊 數據集品質對比'),
            vertical_spacing=0.15
        )
        
        # 1. 按日期和指標的熱力圖
        if len(quality_df) > 0:
            # 準備熱力圖數據
            pivot_data = quality_df.pivot_table(
                index='date', 
                columns='dataset', 
                values='overall_quality', 
                aggfunc='mean'
            ).fillna(0)
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdYlGn',
                    zmin=0, zmax=100,
                    text=[[f'{val:.1f}%' for val in row] for row in pivot_data.values],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="品質評分 (%)")
                ),
                row=1, col=1
            )
        
        # 2. 數據集整體品質對比
        dataset_quality = quality_df.groupby('dataset')['overall_quality'].mean().sort_values(ascending=True)
        
        # 設定顏色
        colors = ['red' if q < 60 else 'orange' if q < 80 else 'green' for q in dataset_quality.values]
        
        fig.add_trace(
            go.Bar(
                x=dataset_quality.values,
                y=dataset_quality.index,
                orientation='h',
                marker_color=colors,
                text=[f'{q:.1f}%' for q in dataset_quality.values],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title={
                'text': '🔥 國道1號圓山-三重路段：數據品質全面分析',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=800,
            template='plotly_white'
        )
        
        # 更新軸標籤
        fig.update_xaxes(title_text="數據集", row=1, col=1)
        fig.update_yaxes(title_text="日期", row=1, col=1)
        fig.update_xaxes(title_text="品質評分 (%)", row=2, col=1)
        fig.update_yaxes(title_text="數據集", row=2, col=1)
        
        # 保存熱力圖
        if save_html:
            output_path = self.output_folder / "data_quality_heatmap.html"
            fig.write_html(str(output_path))
            print(f"   ✅ 數據品質熱力圖已保存: {output_path}")
        
        return fig
    
    def generate_all_visualizations(self):
        """生成所有視覺化圖表"""
        print("🎨 開始生成完整視覺化圖表套組...")
        print("="*60)
        
        generated_files = []
        error_log = []
        
        try:
            # 1. 時間序列分析
            print("📈 生成時間序列分析圖表...")
            try:
                fig1 = self.plot_time_series_analysis()
                if fig1:
                    generated_files.append("time_series_analysis.html")
            except Exception as e:
                error_log.append(f"時間序列分析: {str(e)}")
                print(f"   ❌ 時間序列分析失敗: {e}")
            
            # 2. AI模型推薦
            print("🤖 生成AI模型推薦圖表...")
            try:
                fig2 = self.plot_ai_model_recommendations()
                if fig2:
                    generated_files.append("ai_model_recommendations.html")
                    # 檢查是否有額外的雷達圖
                    radar_file = self.output_folder / "ai_model_radar_chart.html"
                    if radar_file.exists():
                        generated_files.append("ai_model_radar_chart.html")
            except Exception as e:
                error_log.append(f"AI模型推薦: {str(e)}")
                print(f"   ❌ AI模型推薦失敗: {e}")
            
            # 3. 車種分析
            print("🚗 生成車種行為分析圖表...")
            try:
                fig3 = self.plot_vehicle_type_analysis()
                if fig3:
                    generated_files.append("vehicle_type_analysis.html")
            except Exception as e:
                error_log.append(f"車種分析: {str(e)}")
                print(f"   ❌ 車種分析失敗: {e}")
            
            # 4. 數據品質熱力圖
            print("🔥 生成數據品質熱力圖...")
            try:
                fig4 = self.plot_data_quality_heatmap()
                if fig4:
                    generated_files.append("data_quality_heatmap.html")
            except Exception as e:
                error_log.append(f"數據品質熱力圖: {str(e)}")
                print(f"   ❌ 數據品質熱力圖失敗: {e}")
            
            # 5. 互動式儀表板
            print("📊 生成互動式儀表板...")
            try:
                fig5 = self.create_interactive_dashboard()
                if fig5:
                    generated_files.append("interactive_dashboard.html")
            except Exception as e:
                error_log.append(f"互動式儀表板: {str(e)}")
                print(f"   ❌ 互動式儀表板失敗: {e}")
            
            # 生成總結報告（即使有部分失敗也要生成）
            self._generate_visualization_summary(generated_files, error_log)
            generated_files.append("visualization_summary.json")
            
            print("\n🎉 視覺化圖表生成完成！")
            print("="*60)
            print(f"📁 輸出目錄: {self.output_folder}")
            print(f"📊 成功生成: {len(generated_files)} 個檔案")
            
            if error_log:
                print(f"⚠️ 部分失敗: {len(error_log)} 個錯誤")
                for error in error_log:
                    print(f"   • {error}")
            
            for i, filename in enumerate(generated_files, 1):
                print(f"   {i}. {filename}")
            
            print(f"\n🌐 立即查看：")
            if "interactive_dashboard.html" in generated_files:
                print(f"   瀏覽器開啟: {self.output_folder / 'interactive_dashboard.html'}")
            elif generated_files:
                print(f"   瀏覽器開啟: {self.output_folder / generated_files[0]}")
            
            return generated_files
            
        except Exception as e:
            print(f"❌ 視覺化生成過程出錯: {e}")
            # 即使有錯誤也嘗試生成摘要
            try:
                self._generate_visualization_summary(generated_files, [f"總體錯誤: {str(e)}"])
                generated_files.append("visualization_summary.json")
            except:
                pass
            return generated_files
    
    def _generate_visualization_summary(self, generated_files: list, error_log: list = None):
        """生成視覺化摘要報告"""
        
        if error_log is None:
            error_log = []
        
        # 收集統計信息
        total_records = sum(len(df) for df in self.datasets.values())
        dataset_count = len(self.datasets)
        
        # AI分析摘要
        ai_summary = {}
        if 'ai_evaluation' in self.ai_analysis:
            ai_eval = self.ai_analysis['ai_evaluation']
            ai_summary = {
                'lstm_ready': ai_eval['data_readiness'].get('lstm_ready', False),
                'top_model': ai_eval['recommendations'][0]['model'] if ai_eval['recommendations'] else '無',
                'top_score': ai_eval['recommendations'][0]['score'] if ai_eval['recommendations'] else 0,
                'data_quality': self.ai_analysis['data_characteristics']['quality_metrics'].get('overall_quality', 0)
            }
        
        # 創建摘要報告
        summary_report = {
            'visualization_metadata': {
                'generation_time': datetime.now().isoformat(),
                'total_charts': len(generated_files),
                'output_folder': str(self.output_folder),
                'generation_status': 'success' if not error_log else 'partial_success',
                'error_count': len(error_log)
            },
            'data_summary': {
                'total_records': total_records,
                'dataset_count': dataset_count,
                'time_span_days': self.ai_analysis['data_characteristics']['time_coverage'].get('date_span_days', 0) if 'data_characteristics' in self.ai_analysis else 0
            },
            'ai_analysis_summary': ai_summary,
            'generated_files': generated_files,
            'error_log': error_log,
            'recommended_viewing_order': [
                'interactive_dashboard.html - 📊 系統總覽',
                'time_series_analysis.html - 📈 時間序列分析', 
                'ai_model_recommendations.html - 🤖 AI模型推薦',
                'vehicle_type_analysis.html - 🚗 車種行為分析',
                'data_quality_heatmap.html - 🔥 數據品質分析'
            ]
        }
        
        # 保存摘要報告
        summary_path = self.output_folder / "visualization_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📄 視覺化摘要報告已保存: {summary_path}")


# 便利函數
def quick_visualize(base_folder: str = "data") -> list:
    """快速生成所有視覺化圖表"""
    visualizer = TrafficVisualizer(base_folder)
    return visualizer.generate_all_visualizations()


def create_dashboard_only(base_folder: str = "data"):
    """僅生成互動式儀表板"""
    visualizer = TrafficVisualizer(base_folder)
    return visualizer.create_interactive_dashboard()


if __name__ == "__main__":
    print("🎨 啟動交通數據視覺化模組")
    print("="*50)
    print("基於 80,640 筆 AI 訓練數據生成完整視覺化套組")
    print("="*50)
    
    # 創建視覺化器
    visualizer = TrafficVisualizer()
    
    # 檢查數據載入狀況
    if not visualizer.datasets:
        print("❌ 無可用數據，請先執行以下步驟：")
        print("1. python test_loader.py  # 載入原始數據")
        print("2. python test_cleaner.py  # 清理數據")
        print("3. python test_analyzer.py  # 分析數據")
        print("4. 再次執行此視覺化模組")
    else:
        print(f"✅ 數據載入成功，準備生成視覺化圖表...")
        
        # 詢問用戶
        response = input("\n生成完整視覺化套組？(y/N): ")
        
        if response.lower() in ['y', 'yes']:
            # 生成所有圖表
            generated_files = visualizer.generate_all_visualizations()
            
            if generated_files:
                print(f"\n🎉 視覺化完成！共生成 {len(generated_files)} 個圖表")
                print(f"📁 查看位置: {visualizer.output_folder}")
                
                print(f"\n🌐 建議瀏覽順序:")
                print(f"1. 📊 interactive_dashboard.html - 系統總覽儀表板")
                print(f"2. 📈 time_series_analysis.html - 7天時間序列分析")
                print(f"3. 🤖 ai_model_recommendations.html - AI模型智能推薦")
                print(f"4. 🚗 vehicle_type_analysis.html - 車種行為深度分析")
                print(f"5. 🔥 data_quality_heatmap.html - 數據品質熱力圖")
                
                print(f"\n💡 快速查看：")
                print(f"瀏覽器開啟：{visualizer.output_folder / 'interactive_dashboard.html'}")
            else:
                print("❌ 視覺化生成失敗")
        else:
            print("💡 您可以使用以下函數單獨生成圖表：")
            print("   visualizer.create_interactive_dashboard()  # 儀表板")
            print("   visualizer.plot_time_series_analysis()     # 時間序列")
            print("   visualizer.plot_ai_model_recommendations() # AI推薦")
    
    print(f"\n🎯 視覺化模組功能：")
    print("✅ 7天時間序列深度分析")
    print("✅ AI模型智能推薦圖表") 
    print("✅ 車種行為模式分析")
    print("✅ 數據品質全面評估")
    print("✅ 互動式實時儀表板")
    print("✅ 基於80,640筆AI訓練數據")
    
    print(f"\n🚀 下一步建議：")
    print("1. 查看視覺化結果，了解數據模式")
    print("2. 根據AI模型推薦開始模型開發")
    print("3. 使用LSTM進行15分鐘預測建模")
    print("4. 開發 src/predictor.py AI預測模組")