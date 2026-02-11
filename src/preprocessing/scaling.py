import joblib
from sklearn.preprocessing import StandardScaler

def apply_standard_scaling(data, save_path=None):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    if save_path:
        joblib.dump(scaler, save_path)
    return scaled_data, scaler
