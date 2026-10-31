import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)
df=pd.read_csv("data/modelling_table.csv")
X,y=df[FEATURES],df["delivery_hours"]
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42)
scaler=StandardScaler().fit(X_train)
tr_scaled=scaler.transform(X_train); te_scaled=scaler.transform(X_test)
pd.DataFrame([
 {"metric":"train_rows","value":len(X_train)},{"metric":"val_rows","value":len(X_val)},
 {"metric":"test_rows","value":len(X_test)},
 {"metric":"train_scaled_max_abs_mean","value":np.abs(tr_scaled.mean(axis=0)).max()},
 {"metric":"test_scaled_max_abs_mean","value":np.abs(te_scaled.mean(axis=0)).max()}
]).to_csv("outputs/part_b_checks.csv",index=False,float_format="%.4f")

depths=[1,2,3,4,6,8,12,None]; depth_rows=[]
for d in depths:
 m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(X_train,y_train)
 depth_rows.append({"max_depth":"unlimited" if d is None else d,
  "train_rmse":mean_squared_error(y_train,m.predict(X_train))**.5,
  "val_rmse":mean_squared_error(y_val,m.predict(X_val))**.5})
pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv",index=False,float_format="%.4f")
best_depth=min(depth_rows,key=lambda r:(r["val_rmse"], 999 if r["max_depth"]=="unlimited" else r["max_depth"]))["max_depth"]
best_depth=None if best_depth=="unlimited" else int(best_depth)

models={
 "linear_regression":Pipeline([("scaler",StandardScaler()),("model",LinearRegression())]),
 "decision_tree":DecisionTreeRegressor(max_depth=best_depth,random_state=42),
 "random_forest":RandomForestRegressor(n_estimators=100,random_state=42),
 "gradient_boosting":GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42)
}
def metrics(name,split,yt,yp):
 mse=mean_squared_error(yt,yp)
 return {"model":name,"split":split,"MAE":mean_absolute_error(yt,yp),
 "MSE":mse,"RMSE":mse**.5,"R2":r2_score(yt,yp)}
rows=[metrics("mean_baseline","val",y_val,np.repeat(y_train.mean(),len(y_val)))]
for name,m in models.items():
 m.fit(X_train,y_train); rows.append(metrics(name,"val",y_val,m.predict(X_val)))
pd.DataFrame(rows).to_csv("outputs/part_b_metrics.csv",index=False,float_format="%.4f")

lr_rows=[]
for lr in [.01,.1,1.0]:
 m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(X_train,y_train)
 p=m.predict(X_val); lr_rows.append({"learning_rate":lr,"val_rmse":mean_squared_error(y_val,p)**.5,"val_r2":r2_score(y_val,p)})
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv",index=False,float_format="%.4f")

rank=sorted(models,key=lambda n:mean_squared_error(y_val,models[n].predict(X_val)))
cv_rows=[]; cv=KFold(n_splits=5,shuffle=True,random_state=42)
X_tv=pd.concat([X_train,X_val]); y_tv=pd.concat([y_train,y_val])
for name in rank[:2]:
 m=models[name]
 scores=cross_val_score(m,X_tv,y_tv,cv=cv)
 cv_rows.append({"model":name,"mean_r2":scores.mean(),"std_r2":scores.std()})
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv",index=False,float_format="%.4f")
winner=rank[0]
rows.append(metrics(winner,"test",y_test,models[winner].predict(X_test)))
pd.DataFrame(rows).to_csv("outputs/part_b_metrics.csv",index=False,float_format="%.4f")
print("Part B complete. Validation winner:",winner)
