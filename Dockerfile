# Self-contained image for BDD object detection data analysis and training.
# No BDD data is included; mount your dataset at /data when running.
FROM python:3.11-slim

WORKDIR /app

# Install data-analysis deps only (pandas, matplotlib, seaborn, streamlit)
COPY requirements-analysis.txt .
RUN pip install --no-cache-dir -r requirements-analysis.txt

# Copy project: shared data package, data_analysis, top-level entrypoints
COPY data/ data/
COPY data_analysis/ data_analysis/
COPY dashboard.py .
COPY run_analysis.py .

# When running container, mount BDD data at /data and set BDD_ROOT so defaults work
ENV BDD_ROOT=/data
# So that "from data.*" and "from data_analysis.*" resolve when running any script
ENV PYTHONPATH=/app

# Default: run CLI analysis. Override for dashboard or training.
CMD ["python", "run_analysis.py"]
