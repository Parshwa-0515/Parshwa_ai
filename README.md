# 🏏 Cricket AI — Live Win Predictor

IPL T20 in-play win probability model with value bet detection.

## Setup

1. Put all files in one folder:
cricket_ai/
├── app.py
├── requirements.txt
├── matches.csv
├── deliveries.csv
├── batsman.csv
└── bowler.csv

2. Install dependencies:
pip install -r requirements.txt

3. Run locally:
streamlit run app.py

4. Deploy to Streamlit Cloud:
- Push to GitHub
- Connect at streamlit.io
- Deploy!

## Features
- Live win probability (over by over)
- Player-aware predictions (batsman/bowler files)
- Edge detection vs market odds
- Full prediction logging + ROI tracking
- Mobile friendly
