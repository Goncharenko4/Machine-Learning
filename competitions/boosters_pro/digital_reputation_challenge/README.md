## Machine Learning Competition from Digital Reputation
<b> It is necessary to determine the properties of a person's character by its digital footprint:</b> <br>

It is necessary to predict on the presented dataset 5 character properties (1 - property observed, 0 - not observed) of Internet users for more personalized online marketing. <br>
The dataset consists of a digital footprint of people collected from various online sources (for example, the history of site visits, the history of online purchases, etc.). Observations in the presented set are homogeneous, the test and training sets are randomly determined.

### Format of the submission
The submission must be a file format `.csv` with six columns, the first column contains the user id, and the next five contain the predicted observations:
```
id,1,2,3,4,5
59,0.5,0.5,0.5,0.5,0.5
...
```
