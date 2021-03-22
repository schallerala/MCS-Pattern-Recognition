# Pattern Recognition - ex01a
## Alain Schaller - 16-896-375


The first optimization idea, was to reduce the complexity by aggregating the grayscale in regions of different
sizes: dividing the sizes by 14 (with a side of size 2), 7 or 2.

However, the score of the classifier sinked really low, below 20%. 

Therefore, the most effort was done to correctly vectorize the distance operations and see how to process
the train set in parallel.

At the end, the train size is divided equally for each combination of K and distance measure (1500 instances)
and those hyperparameters will then classify only part of the all dataset, to avoid running for hours (tried it
with the original idea, took 8 hours...).

Many improvements could have been applied, however, being new to Python and its API, it had to place most efforts
to overcome its specialities and understand the behavior and structure of the different API. 


### Improvements
As seen in the course, reduce the train set to remove outliers: editing, condensing.



### Result with current state

```text
# precision, K, distance measure
0.9526666666666667, 1,  ManhattanDistanceMeasure
0.9593333333333334, 1,  EuclideanDistanceMeasure
0.9526666666666667, 3,  ManhattanDistanceMeasure
0.9673333333333334, 3,  EuclideanDistanceMeasure
0.9606666666666667, 5,  ManhattanDistanceMeasure
0.9653333333333334, 5,  EuclideanDistanceMeasure
0.9526666666666667, 10, ManhattanDistanceMeasure
0.9553333333333334, 10, EuclideanDistanceMeasure
0.948,              15, ManhattanDistanceMeasure
0.958,              15, EuclideanDistanceMeasure
```
