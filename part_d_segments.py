import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

os.makedirs("outputs",exist_ok=True)
df=pd.read_csv("data/modelling_table.csv")
profile=df.groupby("hub_id").agg(
 shipments=("shipment_id","count"),mean_delivery_hours=("delivery_hours","mean"),
 delay_rate=("is_delayed","mean"),mean_distance_km=("distance_km","mean"),
 mean_n_stops=("n_stops","mean")).reset_index().sort_values("hub_id").reset_index(drop=True)
profile.to_csv("outputs/hub_profile.csv",index=False,float_format="%.4f")
cols=["mean_delivery_hours","delay_rate","mean_distance_km","mean_n_stops"]
scaled=StandardScaler().fit_transform(profile[cols])

sep=[]
for k in [2,3,4,5]:
 labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(scaled)
 within=[]; across=[]
 for i,j in combinations(range(len(profile)),2):
  dist=np.linalg.norm(scaled[i]-scaled[j])
  (within if labels[i]==labels[j] else across).append(dist)
 sizes=sorted(pd.Series(labels).value_counts().tolist(),reverse=True)
 sep.append({"K":k,"across_over_within":np.mean(across)/np.mean(within),
             "cluster_sizes":"-".join(map(str,sizes))})
pd.DataFrame(sep).to_csv("outputs/part_d_separation.csv",index=False,float_format="%.4f")

labels=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(scaled)
profile["cluster"]=labels
means=profile.groupby("cluster").delay_rate.mean().sort_values()
names={cluster:name for cluster,name in zip(means.index,["Standard","Watch","Escalate"])}
profile["segment"]=profile.cluster.map(names)
hubs=pd.read_csv("data/hubs.csv")
segments=profile[["hub_id","segment"]].merge(hubs[["hub_id","hub_city","hub_tier"]],on="hub_id")
segments.to_csv("outputs/hub_segments.csv",index=False)
sp=profile.groupby("segment").agg(hubs=("hub_id","count"),
 mean_delivery_hours=("mean_delivery_hours","mean"),delay_rate=("delay_rate","mean"),
 mean_distance_km=("mean_distance_km","mean"),mean_n_stops=("mean_n_stops","mean")).reset_index()
sp.to_csv("outputs/segment_profile.csv",index=False,float_format="%.4f")
expected={1:"Standard",2:"Watch",3:"Escalate"}
mismatch=segments[segments.apply(lambda r:r.segment!=expected[r.hub_tier],axis=1)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv",index=False)
print("Part D segmentation complete.")
