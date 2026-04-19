# AI-Based Learning Behavior Analysis

An AI-driven project that analyzes student learning and behavioral patterns using Unsupervised Learning (Clustering). This application includes a Streamlit-based dashboard to visualize academic and engagement data, helping identify distinct student groups to provide actionable insights for educators.

## Features
- **Unsupervised Learning**: Groups students into meaningful clusters based on their behavioral, academic, and engagement metrics.
- **Interactive Dashboard**: A sleek, user-friendly Streamlit web interface to explore analytical models.
- **Actionable Insights**: Helps drive data-driven decision-making for educators to support student academic health.

## Project Structure
- `app.py`: The main Streamlit dashboard application.
- `src/`: Core Python modules for the data pipeline:
  - `preprocessing.py`: Code for cleaning, scaling, and preparing the dataset.
  - `model.py`: The machine learning models and clustering algorithms.
  - `visualization.py`: Helper functions for generating rich charts and plots.
- `requirements.txt`: Python package dependencies required to run the project.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pradyumansinghrathore586-design/Ai-based-learning-behavior-analysis.git
   cd Ai-based-learning-behavior-analysis
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   *The app should automatically open in your default web browser at `http://localhost:8501`.*

## License
MIT License
