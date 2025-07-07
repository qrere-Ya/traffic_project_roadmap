# test_imports.py
"""快速測試套件是否正常導入"""

print("🧪 測試套件導入...")

try:
    import numpy as np
    print(f"✅ numpy {np.__version__} - 導入成功")
except Exception as e:
    print(f"❌ numpy導入失敗: {e}")

try:
    import pandas as pd
    print(f"✅ pandas {pd.__version__} - 導入成功")
except Exception as e:
    print(f"❌ pandas導入失敗: {e}")

try:
    import matplotlib
    print(f"✅ matplotlib {matplotlib.__version__} - 導入成功")
except Exception as e:
    print(f"❌ matplotlib導入失敗: {e}")

try:
    import xml.etree.ElementTree as ET
    print("✅ xml.etree.ElementTree - 導入成功")
except Exception as e:
    print(f"❌ xml.etree.ElementTree導入失敗: {e}")

print("\n🎯 如果所有套件都顯示'導入成功'，就可以執行主程式了！")