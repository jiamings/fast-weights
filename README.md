## Using Fast Weights To Attend the Future Past

Reproducing the associative model experiment on the paper [Using Fast Weights To Attend the Future Past](https://arxiv.org/abs/1610.06258) by Jimmy Ba et al. (Incomplete)



### Prerequisites

Tensorflow (version >= 0.8)



### How to Run the Experiments

Generate a dataset

```
$ python generator.py
```

This script generates a file called `associative-retrieval.pkl`, which can be used for training.



Run the model

```
$ python fw.py
```



### Findings

Currently, we are able to see that the accuracy easily exceeds 0.9 for R=20, and 0.97 for R=50, which can justify for the effectiveness of the model. **The experiments are barely tuned.**



**Layer Normalization is extremely crucial for the success of training.** 

- Otherwise, training will not converge when the inner step is larger than 1. 
- Even when inner step of 1, the performance without layer normalization is much worse. For R=20, only 0.4 accuracy can be achieved (which is same as the level of other models.)



Further improvements:

- Complete fine-tuning
- Use accelerated version of A
- Add visualization

