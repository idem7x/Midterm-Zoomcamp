# Midterm-Zoomcamp

## Description of the problem

The online hotel reservation channels have dramatically changed booking possibilities and customers’ behavior. A
significant number of hotel reservations are called-off due to cancellations or no-shows. The typical reasons for
cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free
of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly
revenue-diminishing factor for hotels to deal with.

Model will help to predict if the customer is going to honor the reservation or cancel it.

Used dataset - https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset.
Notebook with all used models - https://www.kaggle.com/code/idem7x/midterm-project and also in **data** folder.

## Instructions on how to run the project

- `git clone git@github.com:idem7x/Midterm-Zoomcamp.git`
- `cd Midterm-Zoomcamp`
- `docker build -t booking_cancellation_predict .`
- `docker run -it --rm -p 9696:9696 booking_cancellation_predict`
- open  http://0.0.0.0:9696

There you will have 2 options to use - RandomForestRegression (button "I want concrete result") and LogisticRegression(
button "I want probability of cancellation")
RandomForestRegression works better on seen data, but it can show wrong result on unseen data.
LogisticRegression showed less accuracy on train data, but it works MUCH better on unseen data.
