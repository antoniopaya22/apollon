from utils import load_data, save_data, load_model
import pandas as pd
from sklearn.model_selection import train_test_split


def merge_data():
    c2017 = load_data(
        ["./data/CIC_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"])
    c2018 = load_data(["./data/CIC_AWS_2018/02-15-2018.csv"])
    c2017.drop([" Fwd Header Length"], axis=1, inplace=True)
    c2018.drop(["Protocol", "Timestamp"], axis=1, inplace=True)
    c2018.columns = c2017.columns
    c_merged = load_data(["./data_prep/merged/CIC-2017_2018.csv"])
    c_merged = pd.concat([c2017, c2018], axis=0)
    c_merged.columns = c2017.columns
    save_data(c_merged, "CIC-2017_2018")


def predict():
    c_merged = load_data(["./data_prep/merged/CIC-2017_2018.csv"])
    x = c_merged.drop([" Label"], axis=1)
    y = c_merged[" Label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print("-----------------")
    model = load_model("../intrusion_detection_systems/models/saved_models/UNSW_dec_tree")
    y_pred = model.predict(x_test)
    print(y_pred)
    

def main():
    predict()
    
if __name__ == "__main__":
    main()
