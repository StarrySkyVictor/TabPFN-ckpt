from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

print('分类：')
# 加载数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# TabPFN
clf_tabpfn = TabPFNClassifier()
clf_tabpfn.fit(X_train, y_train)
pred_tabpfn = clf_tabpfn.predict(X_test)
prob_tabpfn = clf_tabpfn.predict_proba(X_test)
print("\n TabPFN:")
print("Accuracy:", accuracy_score(y_test, pred_tabpfn))
print("ROC AUC (OvR):", roc_auc_score(y_test, prob_tabpfn, multi_class="ovr"))

# Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
pred_rf = clf_rf.predict(X_test)
prob_rf = clf_rf.predict_proba(X_test)
print("\n Random Forest:")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("ROC AUC (OvR):", roc_auc_score(y_test, prob_rf, multi_class="ovr"))

# XGBoost
clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
clf_xgb.fit(X_train, y_train)
pred_xgb = clf_xgb.predict(X_test)
prob_xgb = clf_xgb.predict_proba(X_test)
print("\n XGBoost:")
print("Accuracy:", accuracy_score(y_test, pred_xgb))
print("ROC AUC (OvR):", roc_auc_score(y_test, prob_xgb, multi_class="ovr"))


#---------------------------------------------------------------------------------------------


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

print('回归：')
# 加载数据
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# TabPFN
reg_tabpfn = TabPFNRegressor()
reg_tabpfn.fit(X_train, y_train)
pred_tabpfn = reg_tabpfn.predict(X_test)
print("\n TabPFN Regressor:")
print("MSE:", mean_squared_error(y_test, pred_tabpfn))
print("MAE:", mean_absolute_error(y_test, pred_tabpfn))
print("R²:", r2_score(y_test, pred_tabpfn))

# Random Forest
reg_rf = RandomForestRegressor(random_state=42)
reg_rf.fit(X_train, y_train)
pred_rf = reg_rf.predict(X_test)
print("\n Random Forest Regressor:")
print("MSE:", mean_squared_error(y_test, pred_rf))
print("MAE:", mean_absolute_error(y_test, pred_rf))
print("R²:", r2_score(y_test, pred_rf))

# XGBoost
reg_xgb = XGBRegressor(random_state=42)
reg_xgb.fit(X_train, y_train)
pred_xgb = reg_xgb.predict(X_test)
print("\n XGBoost Regressor:")
print("MSE:", mean_squared_error(y_test, pred_xgb))
print("MAE:", mean_absolute_error(y_test, pred_xgb))
print("R²:", r2_score(y_test, pred_xgb))
