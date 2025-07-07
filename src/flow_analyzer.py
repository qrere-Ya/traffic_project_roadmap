"""
簡化版交通流量探索性數據分析器
========================================

功能：
1. 載入和分析交通數據
2. AI模型適用性評估
3. 智能模型推薦
4. 生成分析報告

作者: 交通預測專案團隊
日期: 2025-07-07 (簡化版)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class SimplifiedTrafficAnalyzer:
    """簡化版交通流量分析器"""
    
    def __init__(self, base_folder: str = "data"):
        self.base_folder = Path(base_folder)
        self.cleaned_folder = self.base_folder / "cleaned"
        self.datasets = {}
        self.analysis_results = {}
        self.insights = []
        
        print("🔬 初始化簡化版交通分析器...")
        self._detect_data_files()
    
    def _detect_data_files(self):
        """檢測可用的數據檔案"""
        if not self.cleaned_folder.exists():
            print(f"❌ 清理數據目錄不存在: {self.cleaned_folder}")
            return
        
        # 檢測日期資料夾
        date_folders = [d for d in self.cleaned_folder.iterdir() 
                       if d.is_dir() and d.name.count('-') == 2]
        
        if date_folders:
            self.available_dates = sorted([d.name for d in date_folders])
            print(f"✅ 發現 {len(self.available_dates)} 個日期資料夾")
        else:
            print("⚠️ 未找到日期資料夾，尋找直接檔案...")
            self.available_dates = []
    
    def load_data(self, merge_dates: bool = True) -> bool:
        """載入數據"""
        print("📊 載入數據...")
        
        try:
            if self.available_dates and merge_dates:
                # 載入多日期數據並合併
                all_data = {}
                
                for date_str in self.available_dates:
                    date_folder = self.cleaned_folder / date_str
                    
                    # 載入目標路段數據（AI訓練主力）
                    peak_file = date_folder / "target_route_peak_cleaned.csv"
                    offpeak_file = date_folder / "target_route_offpeak_cleaned.csv"
                    
                    if peak_file.exists():
                        if 'target_peak' not in all_data:
                            all_data['target_peak'] = []
                        df = pd.read_csv(peak_file)
                        df['source_date'] = date_str
                        all_data['target_peak'].append(df)
                    
                    if offpeak_file.exists():
                        if 'target_offpeak' not in all_data:
                            all_data['target_offpeak'] = []
                        df = pd.read_csv(offpeak_file)
                        df['source_date'] = date_str
                        all_data['target_offpeak'].append(df)
                
                # 合併數據
                for key, df_list in all_data.items():
                    if df_list:
                        self.datasets[key] = pd.concat(df_list, ignore_index=True)
                        print(f"   ✅ {key}: {len(self.datasets[key]):,} 筆記錄")
                
                return len(self.datasets) > 0
            
            else:
                print("❌ 無可用數據")
                return False
                
        except Exception as e:
            print(f"❌ 數據載入失敗: {e}")
            return False
    
    def analyze_data_characteristics(self) -> Dict[str, Any]:
        """分析數據特性"""
        print("🔍 分析數據特性...")
        
        characteristics = {
            'data_summary': {},
            'quality_metrics': {},
            'time_coverage': {},
            'feature_analysis': {}
        }
        
        total_records = 0
        all_dates = set()
        
        for name, df in self.datasets.items():
            if not df.empty:
                # 基本統計
                characteristics['data_summary'][name] = {
                    'records': len(df),
                    'avg_speed': round(df['speed'].mean(), 2) if 'speed' in df.columns else 0,
                    'avg_volume': round(df['volume_total'].mean(), 2) if 'volume_total' in df.columns else 0,
                    'completeness': round(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
                }
                
                total_records += len(df)
                
                # 收集日期
                if 'source_date' in df.columns:
                    all_dates.update(df['source_date'].unique())
        
        # 時間覆蓋分析
        characteristics['time_coverage'] = {
            'total_records': total_records,
            'unique_dates': len(all_dates),
            'date_span_days': len(all_dates),
            'records_per_day': round(total_records / len(all_dates), 0) if all_dates else 0
        }
        
        # 品質評估
        if total_records > 0:
            volume_score = min(100, total_records / 1000)  # 每1000筆記錄得1分
            time_score = min(100, len(all_dates) * 20)      # 每天得20分
            balance_score = self._calculate_balance_score()
            
            overall_quality = (volume_score * 0.4 + time_score * 0.4 + balance_score * 0.2)
            
            characteristics['quality_metrics'] = {
                'volume_score': round(volume_score, 1),
                'time_score': round(time_score, 1),
                'balance_score': round(balance_score, 1),
                'overall_quality': round(overall_quality, 1)
            }
        
        self.analysis_results['data_characteristics'] = characteristics
        
        # 生成洞察
        if total_records > 50000:
            self.insights.append(f"📊 優秀數據規模：{total_records:,}筆記錄適合複雜模型開發")
        if len(all_dates) >= 7:
            self.insights.append(f"📅 理想時間跨度：{len(all_dates)}天數據支援時間序列分析")
        
        return characteristics
    
    def _calculate_balance_score(self) -> float:
        """計算數據平衡性評分"""
        if 'target_peak' in self.datasets and 'target_offpeak' in self.datasets:
            peak_count = len(self.datasets['target_peak'])
            offpeak_count = len(self.datasets['target_offpeak'])
            
            if peak_count > 0 and offpeak_count > 0:
                balance_ratio = min(peak_count, offpeak_count) / max(peak_count, offpeak_count)
                return balance_ratio * 100
        
        return 50  # 預設中等分數
    
    def evaluate_ai_model_suitability(self) -> Dict[str, Any]:
        """評估AI模型適用性"""
        print("🤖 評估AI模型適用性...")
        
        characteristics = self.analysis_results.get('data_characteristics', {})
        time_coverage = characteristics.get('time_coverage', {})
        quality_metrics = characteristics.get('quality_metrics', {})
        
        total_records = time_coverage.get('total_records', 0)
        time_span = time_coverage.get('date_span_days', 0)
        overall_quality = quality_metrics.get('overall_quality', 0)
        
        # 模型適用性評估
        model_suitability = {
            'linear_regression': self._evaluate_linear_model(total_records, overall_quality),
            'random_forest': self._evaluate_random_forest(total_records, overall_quality),
            'xgboost': self._evaluate_xgboost(total_records, overall_quality),
            'lstm': self._evaluate_lstm(total_records, time_span, overall_quality),
            'transformer': self._evaluate_transformer(total_records, time_span, overall_quality),
            'hybrid_ensemble': self._evaluate_hybrid_model(total_records, time_span, overall_quality)
        }
        
        # 生成推薦
        recommendations = self._generate_model_recommendations(model_suitability)
        
        ai_evaluation = {
            'model_suitability': model_suitability,
            'recommendations': recommendations,
            'data_readiness': {
                'records': total_records,
                'time_span': time_span,
                'quality': overall_quality,
                'lstm_ready': total_records >= 50000 and time_span >= 7,
                'production_ready': total_records >= 10000 and overall_quality >= 70
            }
        }
        
        self.analysis_results['ai_evaluation'] = ai_evaluation
        return ai_evaluation
    
    def _evaluate_linear_model(self, records: int, quality: float) -> Dict[str, Any]:
        """評估線性回歸模型"""
        score = min(100, (records / 1000) * 0.5 + quality * 0.5)
        
        return {
            'score': round(score, 1),
            'suitable': records >= 1000,
            'pros': ['快速訓練', '易解釋', '低資源需求'],
            'cons': ['無法捕捉複雜模式', '假設線性關係'],
            'best_for': '基線模型、快速原型'
        }
    
    def _evaluate_random_forest(self, records: int, quality: float) -> Dict[str, Any]:
        """評估隨機森林模型"""
        score = min(100, (records / 5000) * 30 + quality * 0.7)
        
        return {
            'score': round(score, 1),
            'suitable': records >= 5000,
            'pros': ['處理非線性', '特徵重要性', '抗過擬合'],
            'cons': ['記憶體需求大', '預測速度較慢'],
            'best_for': '特徵分析、非線性模式捕捉'
        }
    
    def _evaluate_xgboost(self, records: int, quality: float) -> Dict[str, Any]:
        """評估XGBoost模型"""
        score = min(100, (records / 10000) * 40 + quality * 0.6)
        
        return {
            'score': round(score, 1),
            'suitable': records >= 10000,
            'pros': ['高預測精度', '處理缺失值', '特徵重要性'],
            'cons': ['超參數敏感', '訓練時間較長'],
            'best_for': '高精度預測、結構化數據'
        }
    
    def _evaluate_lstm(self, records: int, time_span: int, quality: float) -> Dict[str, Any]:
        """評估LSTM模型"""
        # LSTM需要充足的時序數據
        time_score = min(100, time_span * 10)  # 每天10分
        data_score = min(100, records / 500)   # 每500筆記錄1分
        score = (time_score * 0.6 + data_score * 0.3 + quality * 0.1)
        
        return {
            'score': round(score, 1),
            'suitable': records >= 50000 and time_span >= 7,
            'pros': ['捕捉時序模式', '長期記憶', '適合預測'],
            'cons': ['訓練時間長', '需要大量數據', 'GPU需求'],
            'best_for': '時間序列預測、長期趨勢分析'
        }
    
    def _evaluate_transformer(self, records: int, time_span: int, quality: float) -> Dict[str, Any]:
        """評估Transformer模型"""
        # Transformer需要更多數據
        time_score = min(100, time_span * 8)
        data_score = min(100, records / 1000)
        score = (time_score * 0.5 + data_score * 0.4 + quality * 0.1)
        
        return {
            'score': round(score, 1),
            'suitable': records >= 100000 and time_span >= 14,
            'pros': ['並行訓練', '注意力機制', '最先進性能'],
            'cons': ['極高資源需求', '需要大量數據', '複雜度高'],
            'best_for': '大規模時序預測、複雜模式識別'
        }
    
    def _evaluate_hybrid_model(self, records: int, time_span: int, quality: float) -> Dict[str, Any]:
        """評估混合集成模型"""
        base_score = (records / 20000) * 40 + (time_span / 14) * 30 + quality * 0.3
        score = min(100, base_score)
        
        return {
            'score': round(score, 1),
            'suitable': records >= 20000 and time_span >= 5,
            'pros': ['結合多模型優勢', '更穩定預測', '風險分散'],
            'cons': ['複雜度高', '訓練時間長', '調參困難'],
            'best_for': '生產環境、高準確率需求'
        }
    
    def _generate_model_recommendations(self, suitability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成模型推薦"""
        # 按評分排序
        scored_models = [(name, info['score'], info['suitable']) 
                        for name, info in suitability.items()]
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        
        for i, (model_name, score, suitable) in enumerate(scored_models):
            if suitable and i < 3:  # 推薦前3個適合的模型
                priority = ['🥇 首選', '🥈 次選', '🥉 備選'][i]
                model_info = suitability[model_name]
                
                recommendations.append({
                    'rank': i + 1,
                    'model': model_name,
                    'priority': priority,
                    'score': score,
                    'reason': model_info['best_for'],
                    'pros': model_info['pros'][:2]  # 只取前兩個優點
                })
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成綜合分析報告"""
        print("📋 生成綜合分析報告...")
        
        # 確保已完成必要分析
        if not self.analysis_results.get('data_characteristics'):
            self.analyze_data_characteristics()
        
        if not self.analysis_results.get('ai_evaluation'):
            self.evaluate_ai_model_suitability()
        
        # 建構報告
        report = {
            'metadata': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analyzer_version': 'simplified_v1.0',
                'total_insights': len(self.insights)
            },
            'data_summary': self.analysis_results['data_characteristics']['data_summary'],
            'quality_assessment': self.analysis_results['data_characteristics']['quality_metrics'],
            'time_coverage': self.analysis_results['data_characteristics']['time_coverage'],
            'ai_model_evaluation': self.analysis_results['ai_evaluation'],
            'key_insights': self.insights,
            'actionable_recommendations': self._generate_actionable_recommendations()
        }
        
        return report
    
    def _generate_actionable_recommendations(self) -> List[str]:
        """生成可行動建議"""
        recommendations = []
        
        ai_eval = self.analysis_results.get('ai_evaluation', {})
        data_readiness = ai_eval.get('data_readiness', {})
        
        total_records = data_readiness.get('records', 0)
        time_span = data_readiness.get('time_span', 0)
        quality = data_readiness.get('quality', 0)
        
        # 基於數據狀況的建議
        if total_records >= 80000:
            recommendations.append("🚀 立即開始深度學習模型開發，數據量充足")
        elif total_records >= 20000:
            recommendations.append("📊 適合開發中等複雜度模型，建議從XGBoost開始")
        else:
            recommendations.append("📈 建議從線性模型和隨機森林開始建立基線")
        
        if time_span >= 7:
            recommendations.append("⏰ 時間序列數據充足，優先考慮LSTM模型")
        else:
            recommendations.append("📅 建議收集更多時間數據以改善時序模型效果")
        
        if quality >= 80:
            recommendations.append("✨ 數據品質優秀，可直接進行模型訓練")
        elif quality >= 70:
            recommendations.append("🔧 數據品質良好，建議進行輕度清理後訓練")
        else:
            recommendations.append("🛠️ 需要進一步改善數據品質")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """保存分析報告"""
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            filename = f"simplified_traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = output_dir / filename
        
        # 確保所有數據都可以JSON序列化
        safe_report = self._safe_json_convert(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(safe_report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 報告已保存: {output_path}")
    
    def _safe_json_convert(self, obj):
        """安全的JSON轉換"""
        if isinstance(obj, dict):
            return {str(k): self._safe_json_convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._safe_json_convert(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def print_summary(self):
        """列印分析摘要"""
        print("\n" + "="*60)
        print("📊 簡化版交通流量分析摘要")
        print("="*60)
        
        # 數據概覽
        if 'data_characteristics' in self.analysis_results:
            time_coverage = self.analysis_results['data_characteristics']['time_coverage']
            quality_metrics = self.analysis_results['data_characteristics']['quality_metrics']
            
            print(f"📈 數據規模:")
            print(f"   總記錄數: {time_coverage.get('total_records', 0):,}")
            print(f"   時間跨度: {time_coverage.get('date_span_days', 0)} 天")
            print(f"   整體品質: {quality_metrics.get('overall_quality', 0)}/100")
        
        # AI模型推薦
        if 'ai_evaluation' in self.analysis_results:
            recommendations = self.analysis_results['ai_evaluation']['recommendations']
            print(f"\n🤖 AI模型推薦:")
            
            for rec in recommendations:
                print(f"   {rec['priority']} {rec['model']}: {rec['score']:.1f}分")
                print(f"      推薦原因: {rec['reason']}")
        
        # 關鍵洞察
        if self.insights:
            print(f"\n💡 關鍵發現:")
            for i, insight in enumerate(self.insights, 1):
                print(f"   {i}. {insight}")
        
        print("\n✅ 簡化版分析完成！")


def quick_analyze(base_folder: str = "data") -> Dict[str, Any]:
    """快速分析函數"""
    analyzer = SimplifiedTrafficAnalyzer(base_folder)
    
    if analyzer.load_data():
        analyzer.analyze_data_characteristics()
        analyzer.evaluate_ai_model_suitability()
        report = analyzer.generate_comprehensive_report()
        analyzer.print_summary()
        analyzer.save_report(report)
        return report
    else:
        print("❌ 數據載入失敗")
        return {}


if __name__ == "__main__":
    print("🚀 啟動簡化版交通流量分析...")
    quick_analyze()