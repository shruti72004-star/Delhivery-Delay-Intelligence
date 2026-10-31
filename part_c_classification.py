import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs",exist_ok=True)
df=pd.read_csv("data/modelling_table.csv"); X=df[FEATURES]; y=df.is_delayed
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,random_state=42,stratify=ytmp)

def score_row(name,yt,yp):
 tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
 return {"model":name,"TN":tn,"FP":fp,"FN":fn,"TP":tp,
 "accuracy":accuracy_score(yt,yp),"precision":precision_score(yt,yp,zero_division=0),
 "recall":recall_score(yt,yp,zero_division=0),"f1":f1_score(yt,yp,zero_division=0)}
rows=[score_row("blind_all_on_time",yv,np.zeros(len(yv),dtype=int))]
log=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000))]).fit(Xtr,ytr)
rows.append(score_row("logistic_regression",yv,log.predict(Xv)))
depth_scores=[]
for d in [2,3,4,5,6,8,10]:
 m=DecisionTreeClassifier(criterion="gini",max_depth=d,random_state=42).fit(Xtr,ytr)
 depth_scores.append((f1_score(yv,m.predict(Xv)),d,m))
_,d,tree=max(depth_scores,key=lambda z:(z[0],-z[1]))
rows.append(score_row(f"decision_tree_depth_{d}",yv,tree.predict(Xv)))
rf_all=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr)
rows.append(score_row("random_forest_default",yv,rf_all.predict(Xv)))
pd.DataFrame(rows).to_csv("outputs/part_c_models.csv",index=False,float_format="%.4f")

selector_all=SelectKBest(f_classif,k="all").fit(Xtr,ytr)
fvals=selector_all.scores_
ranks=pd.Series(fvals,index=FEATURES).rank(method="min",ascending=False).astype(int)
importance=pd.Series(rf_all.feature_importances_,index=FEATURES)
pd.DataFrame([{"feature":f,"f_score":fvals[i],"f_rank":ranks[f],
 "rf_importance":importance[f],"is_noise":f in NOISE_COLUMNS} for i,f in enumerate(FEATURES)]
).to_csv("outputs/part_c_feature_scores.csv",index=False,float_format="%.4f")

krows=[]
for k in [5,7,10,12,17]:
 p=Pipeline([("selector",SelectKBest(f_classif,k=k)),
            ("model",RandomForestClassifier(n_estimators=100,random_state=42))]).fit(Xtr,ytr)
 krows.append({"k":k,"val_f1":f1_score(yv,p.predict(Xv))})
pd.DataFrame(krows).to_csv("outputs/part_c_selectk.csv",index=False,float_format="%.4f")

sel=SelectKBest(f_classif,k=7).fit(Xtr,ytr)
Xt=sel.transform(Xtr); Xval=sel.transform(Xv)
grid_params={"n_estimators":[50,100,150],"max_depth":[4,6,8],"min_samples_leaf":[1,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),grid_params,cv=5).fit(Xt,ytr)
rand=RandomizedSearchCV(RandomForestClassifier(random_state=42),grid_params,n_iter=9,cv=5,random_state=42).fit(Xt,ytr)
default=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xt,ytr)
candidates=[
 ("grid_search",grid.best_params_,grid.best_estimator_,27,5,135,grid.best_score_),
 ("randomized_search",rand.best_params_,rand.best_estimator_,9,5,45,rand.best_score_),
 ("default_forest_same_7_features",{"n_estimators":100,"max_depth":None,"min_samples_leaf":1},default,1,0,1,np.nan)
]
trows=[]
for name,params,m,c,folds,fits,cvacc in candidates:
 trows.append({"search":name,"candidates":c,"cv_folds":folds,"total_fits":fits,
 "best_params":json.dumps(params,sort_keys=True),"best_cv_accuracy":cvacc,
 "val_f1":f1_score(yv,m.predict(Xval))})
pd.DataFrame(trows).to_csv("outputs/part_c_tuning.csv",index=False,float_format="%.4f")

priority={"default_forest_same_7_features":0,"grid_search":1,"randomized_search":2}
winner=max(candidates,key=lambda z:(f1_score(yv,z[2].predict(Xval)),-priority[z[0]]))
chosen_name,params,_,*_=winner
final=Pipeline([("selector",SelectKBest(f_classif,k=7)),
                ("model",RandomForestClassifier(random_state=42,**params))]).fit(Xtr,ytr)
probs=final.predict_proba(Xv)[:,1]
threshold_rows=[]
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
 threshold_rows.append({"threshold":t,**score_row("threshold",yv,(probs>=t).astype(int))})
pd.DataFrame(threshold_rows).drop(columns=["model"]).to_csv("outputs/part_c_thresholds.csv",index=False,float_format="%.4f")
eligible=[r["threshold"] for r in threshold_rows if r["recall"]>=.80]
threshold=max(eligible) if eligible else .1
test_probs=final.predict_proba(Xte)[:,1]
testrow=score_row("final_test",yte,(test_probs>=threshold).astype(int))
pd.DataFrame([{"threshold":threshold,**{k:v for k,v in testrow.items() if k!="model"}}]).to_csv(
 "outputs/part_c_final_test.csv",index=False,float_format="%.4f")
pd.DataFrame([{"selector_k":7,"n_estimators":params["n_estimators"],
 "max_depth":"unlimited" if params.get("max_depth") is None else params.get("max_depth"),
 "min_samples_leaf":params["min_samples_leaf"],"threshold":threshold,"chosen_from":chosen_name}]
).to_csv("outputs/part_c_final_config.csv",index=False)
print("Part C complete. Selected:",chosen_name,"threshold:",threshold)
