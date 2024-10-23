# UniHGKR: Unified Instruction-aware Heterogeneous Knowledge Retrievers

Code and model from our [paper](https://arxiv.org/abs/).

## Abstract

Existing information retrieval (IR) models often assume a homogeneous structure for knowledge sources and user queries, limiting their applicability in real-world settings where retrieval is inherently heterogeneous and diverse.
In this paper, we introduce UniHGKR, a unified instruction-aware heterogeneous knowledge retriever that (1) builds a unified retrieval space for heterogeneous knowledge and (2) follows diverse user instructions to retrieve knowledge of specified types. 
UniHGKR consists of three principal stages: heterogeneous self-supervised pretraining, text-anchored embedding alignment, and instruction-aware retriever fine-tuning, enabling it to generalize across varied retrieval contexts. This framework is highly scalable, with a BERT-based version and a UniHGKR-7B version trained on large language models. 
Also, we introduce CompMix-IR, the first native heterogeneous knowledge retrieval benchmark. It includes two retrieval scenarios with various instructions, over 9,400 question-answer (QA) pairs, and a corpus of 10 million entries, covering four different types of data.
Extensive experiments show that UniHGKR consistently outperforms state-of-the-art methods on CompMix-IR, achieving up to 6.36% and 54.23% relative improvements in two scenarios, respectively.
Finally, by equipping our retriever for open-domain heterogeneous QA systems, we achieve a new state-of-the-art result on the popular [ConvMix](https://convinse.mpi-inf.mpg.de/) task, with an absolute improvement of up to 4.80 points.


## Menu
We are preparing to update more code and benchmark datasets. Please be patient.

### 1. CompMix-IR Benchmark

### 2. UniHGKR model checkpoints

### 3. Code to train and evalutation

#### 3.1 Evalutation on CompMix-IR

#### 3.2 Evalutation on Convmix

#### 3.3 Evalutation on BERI

Our variant model **UniHGKR-base-beir** adapted for evaluation on BEIR can be found at: https://huggingface.co/ZhishanQ/UniHGKR-base-beir

The code for evaluation on BEIR at: [evaluation_beir](https://github.com/ZhishanQ/UniHGKR/tree/main/evaluation_beir).


## Cite
```
temp 
```