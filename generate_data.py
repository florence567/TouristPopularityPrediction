"""Creates dataset/tourism.csv (SYNTHETIC, for demo). Replace with a real CSV with the same columns if you have one."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
dests = {  # name: (state, category, base appeal 0-1, accessibility 1-5)
 "Goa Beaches":("Goa","Beach",.9,5),"Taj Mahal":("Uttar Pradesh","Heritage",.98,5),
 "Jaipur City Palace":("Rajasthan","Heritage",.85,5),"Munnar":("Kerala","Hill Station",.8,3),
 "Manali":("Himachal Pradesh","Hill Station",.85,3),"Varanasi Ghats":("Uttar Pradesh","Spiritual",.85,4),
 "Ladakh":("Ladakh","Adventure",.8,2),"Hampi":("Karnataka","Heritage",.7,3),
 "Rishikesh":("Uttarakhand","Adventure",.8,4),"Andaman Islands":("Andaman","Beach",.75,2),
 "Charminar":("Telangana","Heritage",.7,5),"Ooty":("Tamil Nadu","Hill Station",.75,4),
 "Udaipur":("Rajasthan","Heritage",.8,4),"Kerala Backwaters":("Kerala","Nature",.85,4),
 "Spiti Valley":("Himachal Pradesh","Adventure",.6,1),"Amritsar Golden Temple":("Punjab","Spiritual",.9,5),
 "Darjeeling":("West Bengal","Hill Station",.75,3),"Khajuraho":("Madhya Pradesh","Heritage",.5,2),
 "Ranthambore":("Rajasthan","Wildlife",.6,3),"Mysore Palace":("Karnataka","Heritage",.8,5)}
peak = {"Beach":{"Winter":1,"Summer":.6,"Monsoon":.2},"Hill Station":{"Summer":1,"Winter":.5,"Monsoon":.3},
 "Heritage":{"Winter":1,"Summer":.4,"Monsoon":.5},"Spiritual":{"Winter":.9,"Summer":.6,"Monsoon":.6},
 "Adventure":{"Summer":1,"Winter":.4,"Monsoon":.3},"Nature":{"Monsoon":1,"Winter":.8,"Summer":.5},
 "Wildlife":{"Winter":1,"Summer":.7,"Monsoon":.2}}
temps = {"Summer":(24,40),"Winter":(2,24),"Monsoon":(18,32)}
rows=[]
for _ in range(3000):
    d=rng.choice(list(dests)); st,cat,ap,acc=dests[d]; s=rng.choice(list(temps))
    rating=np.clip(rng.normal(3.3+1.5*ap-.3,.35),1,5).round(1)
    reviews=int(max(20,rng.normal(9000*ap*peak[cat][s]+500,1500)))
    cost=int(np.clip(rng.normal(6000+4000*(1-acc/5)+3000*ap,1500),1500,25000))
    temp=int(rng.integers(*temps[s])); fest=int(rng.random()<.2)
    score=(.35*(rating-1)/4+.25*min(reviews,12000)/12000+.2*peak[cat][s]+.1*acc/5+.1*fest
           -.15*(cost-1500)/23500+.15*ap+rng.normal(0,.05))
    rows.append([d,st,cat,s,rating,reviews,cost,temp,acc,fest,score])
df=pd.DataFrame(rows,columns=["destination","state","category","season","rating","reviews","avg_cost","temperature","accessibility","festival","score"])
q=df.score.quantile([.33,.66]).values
df["popularity"]=pd.cut(df.score,[-9,q[0],q[1],9],labels=["Low","Medium","High"])
df.drop(columns="score").to_csv("dataset/tourism.csv",index=False); print(df.popularity.value_counts())
