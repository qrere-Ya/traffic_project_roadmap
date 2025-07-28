# 國道1號圓山-三重路段交通預測系統

本系統針對國道1號圓山-三重路段實現15分鐘高精度交通預測，結合VD車輛偵測器數據和機器學習技術，為智慧交通管理提供預測支援。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 核心功能

- 即時交通預測：基於歷史數據提供15分鐘精準預測
- 多模型架構：LSTM深度學習、XGBoost梯度提升、隨機森林集成學習
- 智能特徵工程：50+項交通特徵自動化生成
- 高準確率：預測準確率達85%以上
- 實時處理：響應時間小於100毫秒

## 系統架構

### 檔案結構
```
traffic-prediction-system/
├── src/                        
│   ├── data_loader.py          # 數據載入處理
│   ├── data_cleaner.py         # 數據清理模組
│   ├── flow_analyzer.py        # 交通流量分析
│   ├── predictor.py            # 預測模型核心
│   └── visualizer.py           # 數據視覺化
├── data/                       
│   ├── raw/                    # 原始數據
│   ├── processed/              # 處理後數據
│   └── cleaned/                # 清理完成數據
├── models/                     # 訓練完成模型
├── outputs/                    # 輸出結果
├── tests/                      # 測試程式
└── requirements.txt            # 依賴套件
```

### 處理流程
```
數據收集 → 數據清理 → 特徵工程 → 模型訓練 → 預測輸出
```

## 快速開始

### 安裝與環境設定

```bash
git clone https://github.com/username/traffic-prediction-system.git
cd traffic-prediction-system
pip install -r requirements.txt
python test_predictor.py
```

### 基本使用

```python
from src.predictor import TrafficPredictionSystem

# 初始化預測系統
system = TrafficPredictionSystem()

# 訓練模型
system.train_models(data_path="data/cleaned/")

# 執行預測
prediction = system.predict_15_minutes(current_traffic_data)
print(f"預測速度: {prediction['predicted_speed']} km/h")
print(f"交通狀態: {prediction['traffic_status']}")
```

## 研究範圍

### 目標路段
- 起點：圓山交流道 (國道1號23.2K)
- 終點：三重交流道 (國道1號27.0K)
- 總長度：3.8公里雙向路段
- 涵蓋區域：台北市中山區、大同區至新北市三重區

### 關鍵交流道
| 名稱 | 里程位置 | 主要連接道路 |
|------|----------|-------------|
| 圓山交流道 | 23.2K | 建國北路、松江路、濱江街 |
| 台北交流道 | 25.0K | 重慶北路 |
| 三重交流道 | 27.0K | 重陽路、集美街 |

## 數據來源與處理

### 數據來源
- VD車輛偵測器：每分鐘車流量、速度、佔有率數據
- eTag電子標籤：旅行時間、路徑軌跡數據
- 交通管制資訊：事故、施工、管制狀態

### 數據規模
- 訓練數據：80,640筆高品質記錄
- 時間跨度：7天完整週期 (2025-06-21 至 2025-06-27)
- 數據品質：99.8%有效記錄
- 更新頻率：每5分鐘更新

### 數據處理流程
1. 數據載入：XML格式原始數據解析
2. 數據清理：異常值檢測與處理
3. 特徵工程：時間、滯後、滾動統計特徵生成
4. 數據融合：VD與eTag多源數據整合

## 預測模型

### LSTM深度學習模型
- 輸入序列長度：12個時間點 (1小時歷史數據)
- 預測範圍：3個時間點 (15分鐘預測)
- 網路結構：3層LSTM + 全連接層
- 預測準確率：85-92%

### XGBoost梯度提升模型
- 特徵數量：50+ 工程特徵
- 樹深度：8層
- 估計器數量：300棵決策樹
- 預測準確率：80-88%

### 隨機森林模型
- 樹數量：200棵決策樹
- 最大深度：15層
- 特色：基線模型，穩定可靠
- 預測準確率：75-82%

### 特徵工程
- 時間特徵：小時、週期性編碼、尖峰時段識別
- 滯後特徵：1-12期歷史數據
- 滾動統計：3-12期移動平均、標準差
- 交互特徵：速度-密度關係、車種比例

## 系統性能

### 預測性能指標
- RMSE：< 8.0 km/h
- MAE：< 6.0 km/h
- R²：> 0.85
- MAPE：< 12%

### 系統效能
- 響應時間：< 100ms
- 記憶體需求：8GB推薦
- 處理能力：支援百萬級記錄處理
- 可用性：> 99.5%

## 應用場景

### 交通管理應用
- 即時路況監控：提供15分鐘預測視窗
- 事件預警：預測潛在擁堵路段
- 交通調控：智能號誌時制調整建議
- 路線規劃：最佳路徑推薦

### 用戶服務
- 通勤規劃：最佳出行時間建議
- 路況查詢：即時與預測路況資訊
- 導航整合：動態路線規劃支援

## 技術規格

### 系統需求
- Python版本：3.8+
- 最低記憶體：4GB
- 推薦記憶體：8GB+
- 儲存空間：5GB+ (含數據與模型)
- 作業系統：Windows 10+, macOS 10.15+, Ubuntu 18.04+

### 核心依賴
```
tensorflow>=2.13.0
xgboost>=1.7.0
scikit-learn>=1.3.0
pandas>=2.0.3
numpy>=1.24.3
plotly>=5.15.0
```

## 測試與驗證

### 測試覆蓋
```bash
python test_loader.py
python test_cleaner.py
python test_analyzer.py
python test_visualizer.py
python test_predictor.py
```

### 驗證方法
- 時間序列交叉驗證：避免數據洩漏
- 滾動窗口驗證：模擬真實預測場景
- 性能基準測試：響應時間與準確率評估

## 開發指南

### 程式碼結構
- 模組化設計：各功能獨立模組
- 介面標準化：統一的輸入輸出格式
- 錯誤處理：完整的異常處理機制
- 文檔完整：詳細的函數與類別說明

### 擴展開發
- 新數據源整合：支援多種數據格式
- 模型替換：模組化模型架構
- 功能擴展：預留介面便於功能添加

## 授權條款

本專案採用 MIT 授權條款。詳見 [LICENSE](LICENSE) 檔案。

## 聯絡資訊

- 專案維護：[維護者姓名]
- 技術支援：[support@example.com]
- 問題回報：[GitHub Issues](https://github.com/username/traffic-prediction-system/issues)

## 致謝

感謝交通部、高速公路局提供VD偵測器數據，以及所有為智慧交通發展貢獻的研究人員與工程師。
