# A Bayesian Optimization Approach to Machine Translation Reranking [![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](https://arxiv.org/abs/2411.09694)

By Julius Cheng, Maike Züfle, Vilém Zouhar, and Andreas Vlachos.

> Reranking a list of candidates from a machine translation system with an external scoring model and returning the highest-scoring candidate remains a simple and effective method for improving the overall output quality. Translation scoring models continue to grow in size, with the best models being comparable to generation models. Thus, reranking can add substantial computational cost to the translation pipeline. In this work, we pose reranking as a Bayesian optimization (BayesOpt) problem. By strategically selecting candidates to score based on a balance of exploration and exploitation, we show that it is possible to find top-scoring candidates when scoring only a fraction of the candidate list. For instance, our method achieves the same CometKiwi score using only 70 scoring evaluations compared a baseline system using 180. We present a multi-fidelity setting for BayesOpt, where the candidates are first scored with a cheaper but noisier proxy scoring model, which further improves the cost-performance tradeoff when using smaller but well-trained distilled proxy scorers.

<img src="figures/highlevel_schema.svg" height=300em>

## Experiments

TODO


### Training smaller COMET models

The publicly available COMET models are trained with XLM-Roberta, which might be too expensive to run in re-ranking setting.
See `experiments/small_comet` for scripts to train a COMET model with smaller 

## Cite as 
```
@misc{cheng2024bayesianoptimizationapproachmachine,
      title={A Bayesian Optimization Approach to Machine Translation Reranking}, 
      author={Julius Cheng and Maike Züfle and Vilém Zouhar and Andreas Vlachos},
      year={2024},
      eprint={2411.09694},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.09694}, 
}
```
