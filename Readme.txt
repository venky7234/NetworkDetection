# 1. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Convert raw data to CSV (if not already done)
python convert_to_csv.py

# 4. Train the ML model and save it
python train_model.py

# 5. Start the FastAPI backend server
uvicorn backend.app:app --reload

# 6. Open a new terminal for the frontend and run Streamlit
streamlit run frontend/dashboard.py
