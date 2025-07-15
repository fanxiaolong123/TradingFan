#!/usr/bin/env python3
"""
实时监控界面启动脚本

启动增强版的实时监控Web界面
"""

import sys
import os
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """启动实时监控界面"""
    print("🚀 启动 AutoTrader 实时监控界面...")
    print("="*60)
    
    # 检查依赖
    try:
        import streamlit
        print("✅ Streamlit 已安装")
    except ImportError:
        print("❌ Streamlit 未安装，请运行: pip install streamlit")
        return
    
    try:
        import plotly
        print("✅ Plotly 已安装")
    except ImportError:
        print("❌ Plotly 未安装，请运行: pip install plotly")
        return
    
    # 界面路径
    ui_path = project_root / "auto_trader" / "ui" / "realtime_dashboard.py"
    
    if not ui_path.exists():
        print(f"❌ 找不到界面文件: {ui_path}")
        return
    
    print(f"📁 界面文件: {ui_path}")
    print("🌐 启动Web界面...")
    print("="*60)
    print("📌 界面将在浏览器中打开: http://localhost:8501")
    print("📌 使用 Ctrl+C 停止服务")
    print("="*60)
    
    try:
        # 启动streamlit应用
        subprocess.run([
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            str(ui_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=false",
            "--browser.gatherUsageStats=false",
            "--theme.base=light"
        ])
    except KeyboardInterrupt:
        print("\n🛑 界面已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


if __name__ == "__main__":
    main()