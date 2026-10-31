import os
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs",exist_ok=True)
new=pd.read_csv("data/new_shipments.csv")
hubs=pd.read_csv("data/hubs.csv")
med=pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
feature_order=pd.read_csv("models/feature_order.csv")["feature"].tolist()
threshold=float(pd.read_csv("models/delay_threshold.csv").iloc[0,0])
model=joblib.load("models/delay_model.pkl")
clean, _=clean_shipments(new,medians=med)
featured=build_features(clean,hubs)
X=featured[feature_order]
prob=model.predict_proba(X)[:,1]
segments=pd.read_csv("outputs/hub_segments.csv")[["hub_id","segment"]]
out=featured[["shipment_id","hub_id"]].copy()
out["delay_probability"]=prob.round(4)
out["flagged"]=(prob>=threshold).astype(int)
out=out.merge(segments,on="hub_id",how="left",validate="many_to_one")
def action(r):
 if r.flagged==0: return "No action"
 if r.segment=="Escalate": return "Escalate to hub manager"
 return "Proactive customer call"
out["action"]=out.apply(action,axis=1)
out[["shipment_id","hub_id","segment","delay_probability","flagged","action"]].to_csv(
 "outputs/new_shipment_actions.csv",index=False,float_format="%.4f")
print("Predictions complete:",len(out),"shipments")
