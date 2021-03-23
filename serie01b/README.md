# Pattern Recognition - ex01b
## Alain Schaller - 16-896-375

### Improvements
A few steps have been marked as `TODO` in the code to parallelize the work as we do distinct work
on 1 or more clusters.

### Result with the current state

When fitting the model with $\boldsymbol{15'000}$ rows (shifted for each K to get over the entire `train` dataset).

| K  |      C-Index        |      Dunn-Index     |
|:--:|:-------------------:|:-------------------:|
| 5  | 0.27663102832270886 | 0.1606757414632993  |
| 7  | 0.22423321552202627 | 0.10267355040597571 |
| 9  | 0.2369650120372662  | 0.0886003413051528  |
| 10 | 0.19687709629527936 | 0.14939060223923417 |
| 12 | 0.23117394852661927 | 0.09082471309066352 |
| 15 | 0.211636961268018   | 0.09357916169735357 |


When fitting the model with $\boldsymbol{25'000}$ rows (shifted for each K to get over the entire `train` dataset, over a total of $26'997$), which brought my computer to give warning signs of heavy memory usage.
Therefore, won't try higher (really believed my computer would die there).

| K  |      C-Index        |      Dunn-Index     |
|:--:|:-------------------:|:-------------------:|
| 5  | 0.23492856642458593 | 0.1502058595745385  |
| 7  | 0.2247515402325313  | 0.10267355040597571 |
| 9  | 0.23812120649544422 | 0.08611472508724619 |
| 10 | 0.24412236463407777 | 0.08595801159783976 |
| 12 | 0.23596621006702176 | 0.06676618296640102 |
| 15 | 0.23317130713052722 | 0.08732117526706386 |
