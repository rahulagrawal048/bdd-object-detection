"""
Streamlit dashboard for BDD object detection dataset analysis.
Runs the app from data_analysis so that "streamlit run dashboard.py" works from repo root.

Alternatively: streamlit run data_analysis/dashboard.py
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "data_analysis" / "dashboard.py"
    sys.path.insert(0, str(repo_root))
    runpy.run_path(str(app_path), run_name="__main__")
