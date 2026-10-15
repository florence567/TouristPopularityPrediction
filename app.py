import json, joblib, pandas as pd, plotly.express as px, streamlit as st
st.set_page_config(page_title="Tourist Popularity Predictor", page_icon="🧭", layout="wide")

@st.cache_resource
def load(): return joblib.load("model/model.pkl"), json.load(open("model/metrics.json"))
@st.cache_data
def data(): return pd.read_csv("dataset/tourism.csv")
model, m = load(); df = data()

st.title("🧭 Tourist Destination Popularity Predictor")
st.caption("Machine learning (Random Forest) estimates whether a destination will be Low, Medium or High in popularity.")
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Explorer", "🧪 Model Performance"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        dest = st.selectbox("Destination (autofills category)", ["Custom"] + sorted(df.destination.unique()))
        base = df[df.destination == dest].iloc[0] if dest != "Custom" else None
        cats = sorted(df.category.unique())
        cat = st.selectbox("Category", cats, index=cats.index(base.category) if base is not None else 0)
        season = st.selectbox("Season", ["Summer", "Winter", "Monsoon"])
        rating = st.slider("Average rating", 1.0, 5.0, 4.0, .1)
        acc = st.slider("Accessibility (1 = remote, 5 = very easy)", 1, 5, int(base.accessibility) if base is not None else 3)
    with c2:
        reviews = st.number_input("Number of reviews", 0, 50000, 5000, 100)
        cost = st.number_input("Average trip cost (₹)", 1000, 50000, 8000, 500)
        temp = st.slider("Temperature (°C)", -5, 45, 25)
        fest = st.checkbox("Major festival / event during visit")
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([dict(category=cat, season=season, rating=rating, reviews=reviews,
                               avg_cost=cost, temperature=temp, accessibility=acc, festival=int(fest))])
        pred = model.predict(X)[0]; proba = model.predict_proba(X)[0]
        icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[pred]
        st.subheader(f"{icon} Predicted popularity: {pred}")
        pdf = pd.DataFrame({"Popularity": model.classes_, "Probability": proba})
        st.plotly_chart(px.bar(pdf, x="Popularity", y="Probability", range_y=[0, 1], color="Popularity",
                               color_discrete_map={"High": "#2e7d32", "Medium": "#f9a825", "Low": "#c62828"}), use_container_width=True)
        tips = []
        if rating < 3.5: tips.append("Improve visitor experience – the rating is dragging popularity down.")
        if cost > 15000: tips.append("Offer budget packages; high cost reduces demand.")
        if acc <= 2: tips.append("Better transport links would raise footfall.")
        if not fest: tips.append("Hosting a festival or event can boost visits.")
        if reviews < 2000: tips.append("Encourage reviews to increase visibility.")
        st.markdown("**Recommendations**"); [st.write("• " + t) for t in tips or ["Looks strong – maintain quality."]]

with tab2:
    st.write(df.sample(10, random_state=1))
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="category", color="popularity", barmode="group", title="Popularity by category"), use_container_width=True)
    b.plotly_chart(px.histogram(df, x="season", color="popularity", barmode="group", title="Popularity by season"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="rating", y="reviews", color="popularity", title="Rating vs reviews"), use_container_width=True)

with tab3:
    k1, k2 = st.columns(2); k1.metric("Test accuracy", f"{m['accuracy']:.1%}"); k2.metric("5-fold CV accuracy", f"{m['cv_accuracy']:.1%}")
    st.plotly_chart(px.bar(x=list(m["importance"].values()), y=list(m["importance"].keys()), orientation="h",
                           title="Feature importance", labels={"x": "Importance", "y": ""}).update_yaxes(autorange="reversed"), use_container_width=True)
    st.plotly_chart(px.imshow(m["confusion"], x=m["labels"], y=m["labels"], text_auto=True, title="Confusion matrix",
                              labels=dict(x="Predicted", y="Actual")), use_container_width=True)
