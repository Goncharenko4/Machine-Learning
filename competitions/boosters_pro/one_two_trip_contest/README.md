## Competition in machine learning from OneTwoTrip
<b>One had to predict the probability of returning a plane ticket:</b> <br>

If the passenger has changed plans or for some reason he will not be able to use the ticket, he can return it. To do this, he needs to create a refund request. For each order (`order_id`), predict the probability of the situation when user will apply for a refund of the ticket. A .csv file with two columns - the order id (`order_id`) and the probability of return - is accepted as a submission.

Files:
- train.csv - training dataset (the same for both tasks);
- test.csv - test dataset (the same for both tasks);
- baseline1.ipynb - basic solution from the organizers;
- sub1.csv - baseline submit;

Columns in train.csv and test.csv:
- order_id - id of the order for which the probability of ticket refund is predicted;
- user_id - id of the client;
- goal1 - ticket refund (binary variable).

< b>csv files are located here:</b>
https://drive.google.com/drive/folders/14ax74rlqjm0QzWLBvd-AZtYqUobUXfcr?usp=sharing
