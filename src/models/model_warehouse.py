from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


warehouse_dict = {
    "RandomForestClassifier": RandomForestClassifier,
    "XGBoostClassifier": XGBClassifier,
}