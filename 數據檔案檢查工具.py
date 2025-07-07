"""
數據檔案檢查工具
===============

檢查所需的數據檔案是否存在，如果不存在則嘗試創建
"""

import os
import sys
import pandas as pd

def check_files():
    """檢查必要的數據檔案"""
    print("🔍 檢查數據檔案狀況...")
    
    # 檢查原始數據
    raw_files = [
        "data/raw/VD 靜態資訊 (1).xml",
        "data/raw/VD 靜態資訊 (2).xml", 
        "data/raw/VD 靜態資訊 (3).xml",
        "data/raw/VD 靜態資訊 (4).xml",
        "data/raw/VD 靜態資訊 (5).xml"
    ]
    
    # 檢查根目錄的檔案
    root_files = [
        "VD 靜態資訊 (1).xml",
        "VD 靜態資訊 (2).xml",
        "VD 靜態資訊 (3).xml", 
        "VD 靜態資訊 (4).xml",
        "VD 靜態資訊 (5).xml"
    ]
    
    print("\n📁 原始XML檔案:")
    raw_exists = 0
    for file in raw_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
            raw_exists += 1
        else:
            print(f"   ❌ {file}")
    
    print("\n📁 根目錄XML檔案:")
    root_exists = 0
    for file in root_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
            root_exists += 1
        else:
            print(f"   ❌ {file}")
    
    # 檢查處理數據
    processed_files = [
        "data/processed/vd_data_processed.csv",
        "data/processed/vd_data_cleaned.csv"
    ]
    
    print("\n📊 處理後數據:")
    for file in processed_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"   ✅ {file} ({len(df):,} 記錄)")
            except Exception as e:
                print(f"   ⚠️ {file} (無法讀取: {e})")
        else:
            print(f"   ❌ {file}")
    
    return raw_exists, root_exists


def check_columns():
    """檢查數據欄位"""
    print("\n🔍 檢查數據欄位結構...")
    
    # 嘗試不同的檔案路徑
    possible_files = [
        "data/processed/vd_data_cleaned.csv",
        "data/processed/vd_data_processed.csv"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\n📋 {file_path} 的欄位:")
                for i, col in enumerate(df.columns, 1):
                    print(f"   {i:2d}. {col}")
                
                print(f"\n📊 數據摘要:")
                print(f"   記錄數: {len(df):,}")
                print(f"   欄位數: {len(df.columns)}")
                
                # 檢查時間欄位
                time_cols = [col for col in df.columns if 'time' in col.lower()]
                if time_cols:
                    print(f"   時間欄位: {', '.join(time_cols)}")
                else:
                    print("   ⚠️ 未找到時間欄位")
                
                return df.columns.tolist()
                
            except Exception as e:
                print(f"   ❌ 無法讀取 {file_path}: {e}")
    
    print("   ❌ 無法找到可讀取的數據檔案")
    return []


def run_data_loader():
    """嘗試執行數據載入"""
    print("\n🔄 嘗試重新載入數據...")
    
    try:
        # 添加 src 到路徑
        sys.path.append('src')
        from data_loader import VDDataLoader
        
        # 檢查可用的XML檔案
        xml_files = []
        
        # 先檢查根目錄
        for i in range(1, 6):
            file = f"VD 靜態資訊 ({i}).txt"
            if os.path.exists(file):
                xml_files.append(file)
        
        # 再檢查 data/raw 目錄
        for i in range(1, 6):
            file = f"data/raw/VD 靜態資訊 ({i}).txt"
            if os.path.exists(file):
                xml_files.append(file)
        
        if xml_files:
            print(f"   找到 {len(xml_files)} 個XML檔案")
            
            # 確保目錄存在
            os.makedirs("data/processed", exist_ok=True)
            
            # 載入數據
            loader = VDDataLoader()
            df = loader.load_all_files(xml_files)
            
            # 保存處理數據
            output_path = "data/processed/vd_data_processed.csv"
            df.to_csv(output_path, index=False)
            print(f"   ✅ 數據載入成功，保存至: {output_path}")
            print(f"   📊 載入 {len(df):,} 筆記錄")
            
            return True
        else:
            print("   ❌ 找不到XML檔案")
            return False
            
    except Exception as e:
        print(f"   ❌ 數據載入失敗: {e}")
        return False


def run_data_cleaner():
    """嘗試執行數據清理"""
    print("\n🧹 嘗試數據清理...")
    
    try:
        sys.path.append('src')
        from data_cleaner import VDDataCleaner
        
        input_path = "data/processed/vd_data_processed.csv"
        output_path = "data/processed/vd_data_cleaned.csv"
        
        if os.path.exists(input_path):
            cleaner = VDDataCleaner(input_path)
            cleaner.clean_data(method='mark_nan')
            cleaner.save_cleaned_data(output_path)
            
            print(f"   ✅ 數據清理成功，保存至: {output_path}")
            return True
        else:
            print(f"   ❌ 找不到輸入檔案: {input_path}")
            return False
            
    except Exception as e:
        print(f"   ❌ 數據清理失敗: {e}")
        return False


def main():
    """主程序"""
    print("🔧 數據檔案檢查與修復工具")
    print("=" * 50)
    
    # 檢查檔案狀況
    raw_count, root_count = check_files()
    
    # 檢查欄位
    columns = check_columns()
    
    # 如果缺少處理數據，嘗試重新生成
    if not os.path.exists("data/processed/vd_data_cleaned.csv"):
        print("\n🚨 缺少清理後的數據，嘗試重新生成...")
        
        if not os.path.exists("data/processed/vd_data_processed.csv"):
            print("   也缺少原始處理數據，嘗試重新載入...")
            
            if root_count > 0 or raw_count > 0:
                success = run_data_loader()
                if success:
                    run_data_cleaner()
            else:
                print("   ❌ 找不到XML原始檔案")
        else:
            run_data_cleaner()
    
    # 最終檢查
    print("\n" + "=" * 50)
    print("✅ 檢查完成")
    
    if os.path.exists("data/processed/vd_data_cleaned.csv"):
        df = pd.read_csv("data/processed/vd_data_cleaned.csv")
        print(f"📊 可用數據: {len(df):,} 筆記錄")
        print("🚀 可以執行 test_analyzer.py")
    else:
        print("❌ 仍然缺少必要的數據檔案")
        print("💡 請檢查XML原始檔案是否存在")


if __name__ == "__main__":
    main()