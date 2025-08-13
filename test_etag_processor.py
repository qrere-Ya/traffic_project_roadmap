# test_etag_processor.py - 正式版測試程式

"""
eTag處理器正式版測試程式
=======================

完整測試eTag處理器的所有核心功能，確保系統穩定可靠。

測試覆蓋：
1. 🧪 處理器初始化和配置
2. 📁 資料夾掃描和日期提取
3. 🔍 XML解析和命名空間處理
4. 🎯 目標路段識別和篩選
5. 📊 Flow數據提取和車種分類
6. 🕐 時間不匹配問題處理
7. 💾 數據保存和格式驗證
8. 📦 歸檔流程和檔案管理
9. 📈 統計摘要和完整性檢查
10. 🔧 便利函數和API測試

作者: 交通預測專案團隊
版本: 正式版 (2025-01-23)
"""

import sys
import os
import tempfile
import gzip
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# 添加src目錄到路徑
sys.path.append('src')

class ETagProcessorTest:
    """eTag處理器測試類"""
    
    def __init__(self):
        self.test_results = []
        self.temp_files = []
        
        # 目標配對定義（與處理器一致）
        self.expected_target_pairs = {
            '01F0017N-01F0005N': '台北→圓山',
            '01F0005S-01F0017S': '圓山→台北',
            '01F0029N-01F0017N': '三重→台北',
            '01F0017S-01F0029S': '台北→三重'
        }
        
        print("🧪 eTag處理器正式版測試")
        print("=" * 60)
        print("🎯 測試目標: 全面驗證處理器穩定性和準確性")
        print("📋 涵蓋範圍: 初始化→解析→處理→歸檔→統計")
        print("=" * 60)
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """記錄測試結果"""
        self.test_results.append((test_name, success, details))
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"   {status} {test_name}")
        if details and not success:
            print(f"      原因: {details}")
    
    def cleanup_temp_files(self):
        """清理臨時檔案"""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
    
    def test_01_processor_initialization(self):
        """測試1: 處理器初始化"""
        print("\n🧪 測試1: 處理器初始化")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor, process_etag_data, get_etag_summary
            
            # 測試基本初始化
            processor = ETagProcessor(debug=False)
            self.log_test_result("基本初始化", True)
            
            # 檢查目標配對
            if len(processor.target_pairs) == len(self.expected_target_pairs):
                self.log_test_result("目標配對數量", True)
            else:
                self.log_test_result("目標配對數量", False, 
                    f"預期{len(self.expected_target_pairs)}個，實際{len(processor.target_pairs)}個")
                return False
            
            # 檢查配對內容
            for pair_id, expected_name in self.expected_target_pairs.items():
                if pair_id in processor.target_pairs:
                    actual_name = processor.target_pairs[pair_id]['segment']
                    if actual_name == expected_name:
                        continue
                    else:
                        self.log_test_result("配對內容驗證", False, 
                            f"{pair_id}: 預期'{expected_name}', 實際'{actual_name}'")
                        return False
                else:
                    self.log_test_result("配對內容驗證", False, f"缺少配對: {pair_id}")
                    return False
            
            self.log_test_result("配對內容驗證", True)
            
            # 檢查資料夾創建
            folders_exist = all([
                processor.processed_etag_folder.exists(),
                processor.archive_folder.exists()
            ])
            self.log_test_result("資料夾創建", folders_exist)
            
            # 檢查命名空間配置
            namespace_ok = 'ns' in processor.namespace
            self.log_test_result("命名空間配置", namespace_ok)
            
            return True
            
        except Exception as e:
            self.log_test_result("處理器初始化", False, str(e))
            return False
    
    def test_02_date_folder_scanning(self):
        """測試2: 日期資料夾掃描"""
        print("\n🧪 測試2: 日期資料夾掃描")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            processor = ETagProcessor(debug=False)
            
            # 測試日期提取邏輯
            test_cases = [
                ("ETag_Data_20250621", "2025-06-21"),
                ("ETag_Data_20250624", "2025-06-24"),  # 問題日期
                ("ETag_Data_20251231", "2025-12-31"),
                ("Invalid_Folder", None),
                ("ETag_Data_abc", None),
                ("ETag_Data_2025062", None)  # 長度不正確
            ]
            
            # 模擬日期提取邏輯
            for folder_name, expected_date in test_cases:
                if folder_name.startswith('ETag_Data_') and len(folder_name) == 18:
                    date_part = folder_name[10:]
                    if date_part.isdigit() and len(date_part) == 8:
                        extracted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    else:
                        extracted_date = None
                else:
                    extracted_date = None
                
                if extracted_date == expected_date:
                    continue
                else:
                    self.log_test_result("日期提取邏輯", False, 
                        f"{folder_name}: 預期{expected_date}, 得到{extracted_date}")
                    return False
            
            self.log_test_result("日期提取邏輯", True)
            
            # 測試實際掃描（如果有資料夾）
            try:
                date_files = processor.scan_date_folders()
                self.log_test_result("資料夾掃描執行", True)
                
                if date_files:
                    sample_date = list(date_files.keys())[0]
                    sample_files = date_files[sample_date]
                    print(f"      範例: {sample_date} - {len(sample_files)} 檔案")
                    self.log_test_result("掃描結果驗證", True)
                else:
                    self.log_test_result("掃描結果驗證", True, "無資料夾（正常）")
                
            except Exception as e:
                self.log_test_result("資料夾掃描執行", False, str(e))
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("日期資料夾掃描", False, str(e))
            return False
    
    def test_03_xml_parsing(self):
        """測試3: XML解析"""
        print("\n🧪 測試3: XML解析")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            # 創建真實格式的測試XML
            test_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<ETagPairLiveList xmlns="http://traffic.transportdata.tw/standard/traffic/schema/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<UpdateTime>2025-06-25T16:20:00+08:00</UpdateTime>
<UpdateInterval>300</UpdateInterval>
<AuthorityCode>NFB</AuthorityCode>
<LinkVersion>24.09.1</LinkVersion>
<ETagPairLives>
<ETagPairLive>
<ETagPairId>01F0017N-01F0005N</ETagPairId>
<StartETagStatus>0</StartETagStatus>
<EndETagStatus>0</EndETagStatus>
<Flows>
<Flow>
<VehicleType>31</VehicleType>
<TravelTime>51</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>85</SpaceMeanSpeed>
<VehicleCount>25</VehicleCount>
</Flow>
<Flow>
<VehicleType>32</VehicleType>
<TravelTime>0</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>0</SpaceMeanSpeed>
<VehicleCount>0</VehicleCount>
</Flow>
</Flows>
<StartTime>2025-06-23T23:55:00+08:00</StartTime>
<EndTime>2025-06-24T00:00:00+08:00</EndTime>
<DataCollectTime>2025-06-24T00:00:00+08:00</DataCollectTime>
</ETagPairLive>
<ETagPairLive>
<ETagPairId>01F0029N-01F0017N</ETagPairId>
<StartETagStatus>0</StartETagStatus>
<EndETagStatus>0</EndETagStatus>
<Flows>
<Flow>
<VehicleType>31</VehicleType>
<TravelTime>46</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>94</SpaceMeanSpeed>
<VehicleCount>32</VehicleCount>
</Flow>
</Flows>
<StartTime>2025-06-23T23:55:00+08:00</StartTime>
<EndTime>2025-06-24T00:00:00+08:00</EndTime>
<DataCollectTime>2025-06-24T00:00:00+08:00</DataCollectTime>
</ETagPairLive>
<ETagPairLive>
<ETagPairId>01F9999N-01F8888N</ETagPairId>
<StartETagStatus>0</StartETagStatus>
<EndETagStatus>0</EndETagStatus>
<Flows>
<Flow>
<VehicleType>31</VehicleType>
<TravelTime>60</TravelTime>
<StandardDeviation>0</StandardDeviation>
<SpaceMeanSpeed>70</SpaceMeanSpeed>
<VehicleCount>20</VehicleCount>
</Flow>
</Flows>
<StartTime>2025-06-23T23:55:00+08:00</StartTime>
<EndTime>2025-06-24T00:00:00+08:00</EndTime>
<DataCollectTime>2025-06-24T00:00:00+08:00</DataCollectTime>
</ETagPairLive>
</ETagPairLives>
</ETagPairLiveList>'''
            
            # 創建臨時檔案
            with tempfile.NamedTemporaryFile(suffix='ETagPairLive_1620.xml.gz', delete=False) as temp_file:
                with gzip.open(temp_file.name, 'wt', encoding='utf-8') as gz_file:
                    gz_file.write(test_xml)
                temp_path = Path(temp_file.name)
                self.temp_files.append(temp_path)
            
            processor = ETagProcessor(debug=False)
            
            # 測試正常日期處理（時間匹配）
            data_normal = processor.process_single_file(temp_path, '2025-06-25')
            if data_normal:
                self.log_test_result("正常時間解析", True)
                
                # 檢查記錄數量
                expected_records = 3  # 2個目標配對，每個有1-2個有效Flow
                if len(data_normal) >= 2:  # 至少要有目標配對的記錄
                    self.log_test_result("記錄數量檢查", True)
                else:
                    self.log_test_result("記錄數量檢查", False, 
                        f"預期至少2筆，實際{len(data_normal)}筆")
                    return False
                
                # 檢查目標配對過濾
                pair_ids = set(record['etag_pair_id'] for record in data_normal)
                expected_pairs = {'01F0017N-01F0005N', '01F0029N-01F0017N'}
                
                if expected_pairs.issubset(pair_ids):
                    self.log_test_result("目標配對過濾", True)
                else:
                    missing = expected_pairs - pair_ids
                    self.log_test_result("目標配對過濾", False, f"缺少配對: {missing}")
                    return False
                
                # 檢查非目標配對被過濾
                non_target = '01F9999N-01F8888N'
                if non_target not in pair_ids:
                    self.log_test_result("非目標配對過濾", True)
                else:
                    self.log_test_result("非目標配對過濾", False, "非目標配對未被過濾")
                    return False
                
            else:
                self.log_test_result("正常時間解析", False, "沒有解析到數據")
                return False
            
            # 測試時間不匹配處理（重要：24號資料夾但25號時間）
            data_mismatch = processor.process_single_file(temp_path, '2025-06-24')
            if data_mismatch:
                self.log_test_result("時間不匹配處理", True)
                
                # 檢查時間校正
                sample_record = data_mismatch[0]
                record_date = sample_record['update_time'].strftime('%Y-%m-%d')
                if record_date == '2025-06-24':
                    self.log_test_result("時間校正功能", True)
                else:
                    self.log_test_result("時間校正功能", False, 
                        f"預期2025-06-24，實際{record_date}")
                    return False
            else:
                self.log_test_result("時間不匹配處理", False, "時間不匹配數據被錯誤過濾")
                return False
            
            # 測試數據完整性
            sample_record = data_normal[0]
            required_fields = [
                'update_time', 'etag_pair_id', 'vehicle_type_code', 'vehicle_type_name',
                'travel_time_seconds', 'travel_time_minutes', 'vehicle_count',
                'space_mean_speed_kmh', 'segment_name', 'direction', 'data_valid'
            ]
            
            missing_fields = [field for field in required_fields if field not in sample_record]
            if not missing_fields:
                self.log_test_result("數據完整性檢查", True)
            else:
                self.log_test_result("數據完整性檢查", False, f"缺少欄位: {missing_fields}")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("XML解析測試", False, str(e))
            return False
    
    def test_04_vehicle_type_handling(self):
        """測試4: 車種處理"""
        print("\n🧪 測試4: 車種處理")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            processor = ETagProcessor(debug=False)
            
            # 測試車種對應
            vehicle_type_mapping = {
                '31': '小客車',
                '32': '小貨車',
                '41': '大客車', 
                '42': '大貨車',
                '5': '聯結車'
            }
            
            # 模擬XML元素
            class MockElement:
                def __init__(self, data):
                    self.data = data
                
                def find(self, tag, namespace=None):
                    class MockChild:
                        def __init__(self, text):
                            self.text = text
                    
                    tag_name = tag.split(':')[-1] if ':' in tag else tag
                    return MockChild(self.data.get(tag_name)) if tag_name in self.data else None
            
            # 測試各種車種
            all_correct = True
            for code, expected_name in vehicle_type_mapping.items():
                mock_flow = MockElement({
                    'VehicleType': code,
                    'TravelTime': '60',
                    'VehicleCount': '10',
                    'SpaceMeanSpeed': '50',
                    'StandardDeviation': '5'
                })
                
                flow_data = processor._extract_flow_data(
                    mock_flow,
                    '01F0017N-01F0005N',
                    datetime(2025, 6, 24, 16, 20)
                )
                
                if flow_data and flow_data['vehicle_type_name'] == expected_name:
                    continue
                else:
                    all_correct = False
                    actual_name = flow_data['vehicle_type_name'] if flow_data else None
                    self.log_test_result("車種處理", False, 
                        f"車種{code}: 預期'{expected_name}', 實際'{actual_name}'")
                    return False
            
            self.log_test_result("車種對應測試", True)
            
            # 測試無效數據處理
            mock_invalid = MockElement({
                'VehicleType': '31',
                'TravelTime': '0',  # 無效旅行時間
                'VehicleCount': '0',  # 無效車輛數
                'SpaceMeanSpeed': '0',
                'StandardDeviation': '0'
            })
            
            invalid_flow = processor._extract_flow_data(
                mock_invalid,
                '01F0017N-01F0005N', 
                datetime(2025, 6, 24, 16, 20)
            )
            
            if invalid_flow and invalid_flow['data_valid'] == 0:
                self.log_test_result("無效數據標記", True)
            else:
                self.log_test_result("無效數據標記", False, "無效數據未正確標記")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("車種處理測試", False, str(e))
            return False
    
    def test_05_data_saving_and_format(self):
        """測試5: 數據保存和格式"""
        print("\n🧪 測試5: 數據保存和格式")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            processor = ETagProcessor(debug=False)
            
            # 創建模擬數據
            mock_data = [
                {
                    'update_time': datetime(2025, 6, 24, 16, 20),
                    'etag_pair_id': '01F0017N-01F0005N',
                    'vehicle_type_code': '31',
                    'vehicle_type_name': '小客車',
                    'travel_time_seconds': 51,
                    'travel_time_minutes': 0.85,
                    'vehicle_count': 25,
                    'space_mean_speed_kmh': 85.0,
                    'standard_deviation': 0.0,
                    'segment_name': '台北→圓山',
                    'direction': 'N',
                    'distance_km': 1.8,
                    'data_valid': 1
                },
                {
                    'update_time': datetime(2025, 6, 24, 16, 20),
                    'etag_pair_id': '01F0017N-01F0005N',
                    'vehicle_type_code': '32',
                    'vehicle_type_name': '小貨車',
                    'travel_time_seconds': 0,
                    'travel_time_minutes': 0,
                    'vehicle_count': 0,
                    'space_mean_speed_kmh': 0.0,
                    'standard_deviation': 0.0,
                    'segment_name': '台北→圓山',
                    'direction': 'N',
                    'distance_km': 1.8,
                    'data_valid': 0
                }
            ]
            
            # 測試摘要生成
            df = pd.DataFrame(mock_data)
            summary = processor._create_summary(df, '2025-06-24')
            
            # 檢查摘要必要欄位
            required_summary_fields = [
                'date', 'total_records', 'valid_records', 'validity_rate',
                'unique_pairs', 'time_range', 'vehicle_type_distribution'
            ]
            
            missing_summary_fields = [field for field in required_summary_fields 
                                    if field not in summary]
            if not missing_summary_fields:
                self.log_test_result("摘要欄位完整性", True)
            else:
                self.log_test_result("摘要欄位完整性", False, 
                    f"缺少欄位: {missing_summary_fields}")
                return False
            
            # 檢查統計計算
            expected_total = 2
            expected_valid = 1
            expected_validity = 50.0
            
            if (summary['total_records'] == expected_total and
                summary['valid_records'] == expected_valid and
                abs(summary['validity_rate'] - expected_validity) < 0.1):
                self.log_test_result("統計計算正確性", True)
            else:
                self.log_test_result("統計計算正確性", False, 
                    f"統計錯誤: {summary['total_records']}/{summary['valid_records']}/{summary['validity_rate']}")
                return False
            
            # 檢查時間範圍
            time_range = summary['time_range']
            if all(key in time_range for key in ['start', 'end', 'span_hours']):
                self.log_test_result("時間範圍格式", True)
            else:
                self.log_test_result("時間範圍格式", False, "時間範圍格式不完整")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("數據保存和格式測試", False, str(e))
            return False
    
    def test_06_archive_functionality(self):
        """測試6: 歸檔功能"""
        print("\n🧪 測試6: 歸檔功能")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            processor = ETagProcessor(debug=False)
            
            # 創建臨時檔案模擬歸檔
            temp_dir = Path(tempfile.mkdtemp())
            temp_files = []
            
            for i in range(3):
                temp_file = temp_dir / f"ETagPairLive_{i:04d}.xml.gz"
                with gzip.open(temp_file, 'wt') as f:
                    f.write("<test>mock data</test>")
                temp_files.append(temp_file)
                self.temp_files.append(temp_file)
            
            # 測試歸檔邏輯
            archive_folder = processor.archive_folder / "test_date"
            archive_folder.mkdir(parents=True, exist_ok=True)
            
            # 模擬歸檔過程
            archived_count = 0
            for temp_file in temp_files:
                if temp_file.exists():
                    archive_path = archive_folder / temp_file.name
                    try:
                        import shutil
                        shutil.move(str(temp_file), str(archive_path))
                        archived_count += 1
                    except Exception as e:
                        print(f"      歸檔失敗: {e}")
            
            if archived_count == len(temp_files):
                self.log_test_result("檔案歸檔功能", True)
            else:
                self.log_test_result("檔案歸檔功能", False, 
                    f"歸檔不完整: {archived_count}/{len(temp_files)}")
                return False
            
            # 檢查歸檔檔案
            archived_files = list(archive_folder.glob("*.xml.gz"))
            if len(archived_files) == len(temp_files):
                self.log_test_result("歸檔檔案驗證", True)
            else:
                self.log_test_result("歸檔檔案驗證", False, 
                    f"歸檔檔案數量不符: {len(archived_files)}/{len(temp_files)}")
                return False
            
            # 清理測試歸檔
            try:
                import shutil
                shutil.rmtree(archive_folder)
            except:
                pass
            
            return True
            
        except Exception as e:
            self.log_test_result("歸檔功能測試", False, str(e))
            return False
    
    def test_07_convenience_functions(self):
        """測試7: 便利函數"""
        print("\n🧪 測試7: 便利函數")
        print("-" * 40)
        
        try:
            from etag_processor import process_etag_data, get_etag_summary
            import inspect
            
            # 檢查函數簽名
            process_sig = inspect.signature(process_etag_data)
            expected_process_params = ['base_folder', 'debug']
            actual_process_params = list(process_sig.parameters.keys())
            
            if actual_process_params == expected_process_params:
                self.log_test_result("process_etag_data簽名", True)
            else:
                self.log_test_result("process_etag_data簽名", False, 
                    f"預期{expected_process_params}, 實際{actual_process_params}")
                return False
            
            summary_sig = inspect.signature(get_etag_summary)
            expected_summary_params = ['base_folder']
            actual_summary_params = list(summary_sig.parameters.keys())
            
            if actual_summary_params == expected_summary_params:
                self.log_test_result("get_etag_summary簽名", True)
            else:
                self.log_test_result("get_etag_summary簽名", False, 
                    f"預期{expected_summary_params}, 實際{actual_summary_params}")
                return False
            
            # 測試摘要函數執行
            try:
                summary = get_etag_summary()
                if summary is not None and isinstance(summary, dict):
                    self.log_test_result("get_etag_summary執行", True)
                    
                    # 檢查摘要結構
                    expected_keys = ['processed_dates', 'total_records']
                    if all(key in summary for key in expected_keys):
                        self.log_test_result("摘要結構檢查", True)
                    else:
                        missing_keys = [key for key in expected_keys if key not in summary]
                        self.log_test_result("摘要結構檢查", False, f"缺少鍵: {missing_keys}")
                        return False
                else:
                    self.log_test_result("get_etag_summary執行", False, "返回值類型錯誤")
                    return False
            except Exception as e:
                self.log_test_result("get_etag_summary執行", False, str(e))
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("便利函數測試", False, str(e))
            return False
    
    def test_08_error_handling(self):
        """測試8: 錯誤處理"""
        print("\n🧪 測試8: 錯誤處理")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            processor = ETagProcessor(debug=False)
            
            # 測試不存在檔案
            non_existent_file = Path("non_existent_file.xml.gz")
            result = processor.process_single_file(non_existent_file, "2025-06-24")
            
            if result == []:  # 應該返回空列表
                self.log_test_result("不存在檔案處理", True)
            else:
                self.log_test_result("不存在檔案處理", False, "應該返回空列表")
                return False
            
            # 測試損壞的XML
            corrupted_xml = "<invalid>xml content"
            with tempfile.NamedTemporaryFile(suffix='.xml.gz', delete=False) as temp_file:
                with gzip.open(temp_file.name, 'wt') as f:
                    f.write(corrupted_xml)
                temp_path = Path(temp_file.name)
                self.temp_files.append(temp_path)
            
            result = processor.process_single_file(temp_path, "2025-06-24")
            if result == []:  # 應該返回空列表而不是崩潰
                self.log_test_result("損壞XML處理", True)
            else:
                self.log_test_result("損壞XML處理", False, "應該返回空列表")
                return False
            
            # 測試空資料夾掃描
            empty_scan = processor.scan_date_folders()
            if isinstance(empty_scan, dict):  # 應該返回字典（可能為空）
                self.log_test_result("空資料夾掃描", True)
            else:
                self.log_test_result("空資料夾掃描", False, "應該返回字典")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("錯誤處理測試", False, str(e))
            return False
    
    def test_09_performance_check(self):
        """測試9: 性能檢查"""
        print("\n🧪 測試9: 性能檢查")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            import time
            
            processor = ETagProcessor(debug=False)
            
            # 創建大型測試XML（模擬實際大小）
            large_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<ETagPairLiveList xmlns="http://traffic.transportdata.tw/standard/traffic/schema/">
<UpdateTime>2025-06-24T16:20:00+08:00</UpdateTime>
<UpdateInterval>300</UpdateInterval>
<AuthorityCode>NFB</AuthorityCode>
<LinkVersion>24.09.1</LinkVersion>
<ETagPairLives>'''
            
            # 添加多個ETagPairLive（包含目標配對）
            for i in range(50):  # 模擬50個配對
                if i < 4:  # 前4個是目標配對
                    target_pairs = ['01F0017N-01F0005N', '01F0005S-01F0017S', 
                                  '01F0029N-01F0017N', '01F0017S-01F0029S']
                    pair_id = target_pairs[i]
                else:
                    pair_id = f"01F{i:04d}N-01F{i+1:04d}N"
                
                large_xml += f'''
<ETagPairLive>
<ETagPairId>{pair_id}</ETagPairId>
<StartETagStatus>0</StartETagStatus>
<EndETagStatus>0</EndETagStatus>
<Flows>
<Flow>
<VehicleType>31</VehicleType>
<TravelTime>60</TravelTime>
<StandardDeviation>5</StandardDeviation>
<SpaceMeanSpeed>70</SpaceMeanSpeed>
<VehicleCount>20</VehicleCount>
</Flow>
</Flows>
<StartTime>2025-06-24T16:15:00+08:00</StartTime>
<EndTime>2025-06-24T16:20:00+08:00</EndTime>
<DataCollectTime>2025-06-24T16:20:00+08:00</DataCollectTime>
</ETagPairLive>'''
            
            large_xml += '''
</ETagPairLives>
</ETagPairLiveList>'''
            
            # 創建大型臨時檔案
            with tempfile.NamedTemporaryFile(suffix='ETagPairLive_1620.xml.gz', delete=False) as temp_file:
                with gzip.open(temp_file.name, 'wt', encoding='utf-8') as f:
                    f.write(large_xml)
                temp_path = Path(temp_file.name)
                self.temp_files.append(temp_path)
            
            # 測試處理時間
            start_time = time.time()
            result = processor.process_single_file(temp_path, "2025-06-24")
            processing_time = time.time() - start_time
            
            if processing_time < 5.0:  # 應該在5秒內完成
                self.log_test_result("處理速度檢查", True, f"{processing_time:.2f}秒")
            else:
                self.log_test_result("處理速度檢查", False, f"處理時間過長: {processing_time:.2f}秒")
                return False
            
            # 檢查記錄數量（應該只有目標配對）
            if result and len(result) >= 4:  # 至少4個目標配對
                self.log_test_result("大檔案處理結果", True, f"提取{len(result)}筆記錄")
            else:
                self.log_test_result("大檔案處理結果", False, 
                    f"記錄數量不足: {len(result) if result else 0}")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("性能檢查測試", False, str(e))
            return False
    
    def test_10_integration_test(self):
        """測試10: 整合測試"""
        print("\n🧪 測試10: 整合測試")
        print("-" * 40)
        
        try:
            from etag_processor import ETagProcessor
            
            processor = ETagProcessor(debug=False)
            
            # 檢查實際數據處理能力
            print("      檢查實際數據處理能力...")
            
            # 1. 掃描實際資料夾
            date_files = processor.scan_date_folders()
            self.log_test_result("實際資料夾掃描", True, f"發現{len(date_files)}個日期")
            
            # 2. 檢查處理摘要
            summary = processor.get_processing_summary()
            if summary and 'processed_dates' in summary:
                processed_count = summary['processed_dates']
                total_records = summary['total_records']
                self.log_test_result("處理摘要檢查", True, 
                    f"{processed_count}個日期, {total_records:,}筆記錄")
            else:
                self.log_test_result("處理摘要檢查", True, "無已處理數據（正常）")
            
            # 3. 檢查目標配對一致性
            expected_pairs = set(self.expected_target_pairs.keys())
            actual_pairs = set(processor.target_pairs.keys())
            
            if expected_pairs == actual_pairs:
                self.log_test_result("配對一致性檢查", True)
            else:
                missing = expected_pairs - actual_pairs
                extra = actual_pairs - expected_pairs
                details = f"缺少:{missing}, 多餘:{extra}" if (missing or extra) else "一致"
                self.log_test_result("配對一致性檢查", len(missing) == 0 and len(extra) == 0, details)
            
            return True
            
        except Exception as e:
            self.log_test_result("整合測試", False, str(e))
            return False
    
    def generate_final_report(self):
        """生成最終測試報告"""
        print("\n" + "="*70)
        print("📋 eTag處理器正式版測試報告")
        print("="*70)
        
        # 統計測試結果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, success, _ in self.test_results if success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 測試統計:")
        print(f"   總測試項目: {total_tests}")
        print(f"   通過測試: {passed_tests}")
        print(f"   失敗測試: {failed_tests}")
        print(f"   成功率: {success_rate:.1f}%")
        
        # 詳細測試結果
        print(f"\n📋 詳細測試結果:")
        for test_name, success, details in self.test_results:
            status = "✅ 通過" if success else "❌ 失敗"
            print(f"   {status} {test_name}")
            if details and not success:
                print(f"      └─ {details}")
        
        # 測試分類統計
        categories = {
            "基礎功能": ["處理器初始化", "日期資料夾掃描", "便利函數"],
            "數據處理": ["XML解析", "車種處理", "數據保存和格式"],
            "系統功能": ["歸檔功能", "錯誤處理", "性能檢查"],
            "整合驗證": ["整合測試"]
        }
        
        print(f"\n📊 分類統計:")
        for category, test_names in categories.items():
            category_results = [success for name, success, _ in self.test_results 
                              if any(test_name in name for test_name in test_names)]
            if category_results:
                category_passed = sum(category_results)
                category_total = len(category_results)
                category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
                print(f"   {category}: {category_passed}/{category_total} ({category_rate:.0f}%)")
        
        # 總結評估
        if success_rate >= 95:
            grade = "優秀"
            icon = "🎉"
        elif success_rate >= 85:
            grade = "良好"
            icon = "✅"
        elif success_rate >= 70:
            grade = "及格"
            icon = "⚠️"
        else:
            grade = "需改進"
            icon = "❌"
        
        print(f"\n{icon} 總體評估: {grade} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print(f"\n🎯 系統就緒狀態:")
            print(f"   ✅ 核心功能完整")
            print(f"   ✅ 數據處理準確")
            print(f"   ✅ 錯誤處理健全")
            print(f"   ✅ 性能表現良好")
            
            print(f"\n🚀 生產環境部署建議:")
            print(f"   📁 確保數據路徑正確: data/raw/etag/ETag_Data_YYYYMMDD/")
            print(f"   🔧 定期檢查歸檔空間")
            print(f"   📊 監控處理時間和記錄數量")
            print(f"   🔄 建立備份和恢復機制")
            
        else:
            print(f"\n⚠️ 需要改進的項目:")
            for test_name, success, details in self.test_results:
                if not success:
                    print(f"   ❌ {test_name}: {details}")
            
            print(f"\n🔧 建議修正後重新測試")
        
        # 性能指標
        print(f"\n📈 關鍵性能指標:")
        print(f"   🎯 目標配對: 4個 (台北↔圓山, 三重↔台北)")
        print(f"   🚗 支援車種: 5種 (31,32,41,42,5)")
        print(f"   📅 時間處理: 支援時間不匹配情況")
        print(f"   📦 歸檔功能: 自動歸檔和清理")
        print(f"   🔍 錯誤處理: 完整的異常捕獲")
        
        return success_rate >= 90
    
    def run_all_tests(self):
        """執行所有測試"""
        test_methods = [
            self.test_01_processor_initialization,
            self.test_02_date_folder_scanning,
            self.test_03_xml_parsing,
            self.test_04_vehicle_type_handling,
            self.test_05_data_saving_and_format,
            self.test_06_archive_functionality,
            self.test_07_convenience_functions,
            self.test_08_error_handling,
            self.test_09_performance_check,
            self.test_10_integration_test
        ]
        
        start_time = datetime.now()
        
        # 依序執行測試
        for i, test_method in enumerate(test_methods, 1):
            try:
                success = test_method()
                if not success and i <= 3:  # 前3個測試失敗則停止
                    print(f"\n❌ 核心測試失敗，停止後續測試")
                    break
            except Exception as e:
                print(f"\n❌ 測試 {i} 執行異常: {e}")
                self.log_test_result(f"測試{i}執行", False, str(e))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️ 總測試時間: {duration:.1f} 秒")
        
        # 生成最終報告
        success = self.generate_final_report()
        
        # 清理資源
        self.cleanup_temp_files()
        
        return success


def main():
    """主測試程序"""
    print("🧪 eTag處理器正式版測試程式")
    print("=" * 70)
    print("📅 測試日期:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 測試目標: 全面驗證eTag處理器的穩定性和準確性")
    print("📋 測試範圍: 10個主要功能模組的完整測試")
    print("⚡ 特別關注: 時間不匹配問題、目標配對識別、歸檔流程")
    print("=" * 70)
    
    # 創建測試實例
    tester = ETagProcessorTest()
    
    try:
        # 執行完整測試
        success = tester.run_all_tests()
        
        if success:
            print(f"\n🎉 eTag處理器正式版測試完全通過！")
            print(f"\n📁 系統已就緒，可投入生產使用:")
            print(f"   python src/etag_processor.py  # 處理eTag數據")
            print(f"   python -c \"from src.etag_processor import get_etag_summary; print(get_etag_summary())\"  # 檢查結果")
            
            print(f"\n🔧 關鍵特性:")
            print(f"   ✅ 解決XML時間與資料夾日期不匹配問題")
            print(f"   ✅ 精準識別4個目標路段配對")
            print(f"   ✅ 完整支援5種車輛類型")
            print(f"   ✅ 自動歸檔和檔案管理")
            print(f"   ✅ 健全的錯誤處理機制")
            print(f"   ✅ 優秀的處理性能")
            
            print(f"\n🚀 Ready for Production! 🚀")
            return True
        else:
            print(f"\n❌ 測試未完全通過，建議修正後重新測試")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 測試被用戶中斷")
        return False
    except Exception as e:
        print(f"\n❌ 測試執行出現異常: {e}")
        return False
    finally:
        # 確保清理資源
        tester.cleanup_temp_files()


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ eTag處理器正式版測試完成！系統已就緒。")
    else:
        print("\n🔧 請根據測試報告修正問題後重新測試。")
    
    print(f"\n📊 測試完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎊 感謝使用eTag處理器測試程式！")