# GRU4Rec Sequential Recommender System 

This repository contains the implementation, optimization, and evaluation of a session-based recommender system using the **GRU4Rec** architecture. 

* **Dataset:** [Yoochoose / RecSys Challenge 2015](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) (33M+ interactions).
* **Reference Implementation:** Adapted and hot-patched from the [Official GRU4Rec PyTorch Repository](https://github.com/hidasib/GRU4Rec_PyTorch_Official).

## Project Overview
The entire pipeline was engineered to run in a resource-constrained environment (Google Colab) by applying strict memory management techniques and dynamic hot-patching of legacy repository code.

### Key Engineering Decisions:
1. **Memory Optimization (Data Engineering):**
   * Avoided the standard practice of truncating the dataset to 1/8. Instead, the full dataset was processed by explicitly downcasting numeric types from `int64` to `int32` upon reading.
   * Filtered noise (items with < 5 interactions, sessions with < 2 clicks) and actively managed RAM using explicit garbage collection.
2. **Runtime Hot-Patching:**
   * The underlying official GRU4Rec PyTorch repository contained legacy bugs regarding tensor data types (`float64` vs `float32` mismatches).
   * Developed an automated runtime patching script that rewrites the tensor initializations directly in the source code before execution, ensuring environment reproducibility.
3. **Strict Chronological Data Splitting:**
   * To prevent data leakage in the time-series context, the dataset was split strictly sequentially (90% Train, 10% Test) without random shuffling.

## Final Performance Metrics
The model was trained using the BPR-max loss function and Adagrad optimizer. Evaluation was performed on completely unseen data.

* **Recall@20:** `0.6733`
* **MRR@20:** `0.2997`

> **Performance vs. Optimization Note:** 
> Despite implementing aggressive memory optimization techniques (such as `int32` downcasting and filtering) to process the entire dataset in a constrained environment, the model's accuracy remained nearly identical to the benchmarks reported in the [official repository](https://github.com/hidasib/GRU4Rec_PyTorch_Official) (Official Recall@20: 0.6789 | Official MRR@20: 0.3024). This confirms that the architectural optimizations successfully reduced memory footprint without sacrificing predictive power.

## Repository Structure
* `GRU4Rec_Yoochoose_Pipeline.ipynb`: The complete end-to-end Jupyter Notebook containing environment setup, bug-fixing patches, data processing, training execution, and evaluation.
