import pandas as pd
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


def predict_subscribe(watch, duration, ctr, interest):
    data = [{"watch_time": watch, "avg_view_duration": duration, "click_through_rate": ctr, "interest": interest}]
    sample = pd.DataFrame(data)
    pred = model.predict_proba(sample.values)[0]
    return {'Misses Out': pred[0], 'Subscribes': pred[1]}