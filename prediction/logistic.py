import pickle

with open('../bin/model-logistic.bin', 'rb') as f_model:
    model = pickle.load(f_model)

with open('../bin/dv-logistic.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

client = {
"no_of_adults": 2,
"no_of_children":  0,
"no_of_weekend_nights": 1,
"no_of_week_nights": 2,
"type_of_meal_plan": "meal_plan_1",
"required_car_parking_space": 0,
"room_type_reserved": "room_type_1",
"lead_time": 224,
"arrival_year": 0,
"arrival_month": 10,
"arrival_date": 2,
"market_segment_type": "offline",
"repeated_guest": 0,
"no_of_previous_cancellations": 0,
"no_of_previous_bookings_not_canceled": 0,
"avg_price_per_room": 65,
"no_of_special_requests": 0
}

bad_client = {
"no_of_adults": 2,
"no_of_children":  0,
"no_of_weekend_nights": 0,
"no_of_week_nights": 2,
"type_of_meal_plan": "meal_plan_1",
"required_car_parking_space": 0,
"room_type_reserved": "room_type_1",
"lead_time": 211,
"arrival_year": 1,
"arrival_month": 5,
"arrival_date": 20,
"market_segment_type": "online",
"repeated_guest": 0,
"no_of_previous_cancellations": 0,
"no_of_previous_bookings_not_canceled": 0,
"avg_price_per_room": 100,
"no_of_special_requests": 0
}

X = dv.transform([bad_client])
print(X)
result = model.predict_proba(X)[0, 1]
# rounded_result = round(result, 3)
print(result)
