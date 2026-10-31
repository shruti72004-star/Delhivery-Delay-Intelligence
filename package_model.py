import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models",exist_ok=True); os.makedirs("outputs",exist_ok=True)
df=pd.read_csv("data/modelling_table.csv")
X,y=df[FEATURES],df.is_delayed
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)
cfg=pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
depth=None if str(cfg.max_depth)=="unlimited" else int(cfg.max_depth)
model=Pipeline([("selector",SelectKBest(f_classif,k=int(cfg.selector_k))),
 ("model",RandomForestClassifier(n_estimators=int(cfg.n_estimators),max_depth=depth,
 min_samples_leaf=int(cfg.min_samples_leaf),random_state=42))]).fit(Xtr,ytr)
joblib.dump(model,"models/delay_model.pkl")
pd.DataFrame({"threshold":[float(cfg.threshold)]}).to_csv("models/delay_threshold.csv",index=False)
pd.DataFrame({"feature":FEATURES}).to_csv("models/feature_order.csv",index=False)
raw=pd.read_csv("data/raw_shipments.csv")
_,medians=clean_shipments(raw)
pd.DataFrame([{"column":k,"median":v} for k,v in medians.items()]).to_csv("models/impute_values.csv",index=False)
loaded=joblib.load("models/delay_model.pkl")
diff=np.max(np.abs(model.predict_proba(Xte)[:,1]-loaded.predict_proba(Xte)[:,1]))
pd.DataFrame([{"metric":"test_rows_scored","value":len(Xte)},
 {"metric":"reload_max_abs_probability_diff","value":diff},
 {"metric":"threshold","value":float(cfg.threshold)}]).to_csv("outputs/part_d_reload_check.csv",index=False,float_format="%.10f")
print("Model packaged. Reload probability difference:",diff)
