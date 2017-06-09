# CIFAR experiments

The code requires Lasagne (version 0.1). Instructions for installation:
http://lasagne.readthedocs.io/en/latest/user/installation.html 

## Training

Train the MIX+DCGAN:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_mixgan.py
```

## Evaluation

Evaluate the Inception Score of the MIX+DCGAN:

First, sample:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sampling_mixgan.py
```
Second, evaluate:

```
python eval_mixgan.py
```



