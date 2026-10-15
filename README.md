# Tourist Destination Popularity Prediction
Streamlit + scikit-learn app predicting Low / Medium / High popularity of a destination.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python generate_data.py        # skip if you use your own real CSV
python explore.py
python train.py
streamlit run app.py
```
## Using a real dataset
Put your CSV at `dataset/tourism.csv` with columns: destination, state, category, season, rating, reviews, avg_cost, temperature, accessibility, festival, popularity (Low/Medium/High). If yours differ, edit `CAT`/`NUM` in `train.py` and the inputs in `app.py`.

## Deploy
Push the folder (incl. `model/`, `dataset/`) to GitHub → share.streamlit.io → select repo → main file `app.py`.
