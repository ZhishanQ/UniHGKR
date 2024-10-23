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

**The corpus of CompMix-IR:**
1. Google Drive: [Link](https://drive.google.com/file/d/1sDmPieBkAnO9Rb7oDDXAgRDd5SRo_rPP/view?usp=sharing)
2. HuggingFace Dataset: [Link](https://huggingface.co/datasets/ZhishanQ/CompMix-IR)

**The QA pairs of CompMix:**

CompMix QA pairs: [CompMix](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/CompMix)

ConvMix QA pairs: [ConvMix_annotated](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/ConvMix_annotated)

or Huggingface dataset:

[CompMix](https://huggingface.co/datasets/pchristm/CompMix)

[ConvMix](https://huggingface.co/datasets/pchristm/ConvMix)

Code to evaluate whether the retrieved evidence is relevant to the question:

[Code to judge relevance](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/eval_part)


### 2. UniHGKR model checkpoints
 
| Mdeol Name            | Description                                                                                                                | Huggingface  Link                                                              | Usage Example                                                                                                         |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| UniHGKR-base          | adapted for evaluation on CompMix-IR                                                                                       | [UniHGKR-base](https://huggingface.co/ZhishanQ/UniHGKR-base)                   | [demo code to use](https://github.com/ZhishanQ/UniHGKR/tree/main/code_for_UniHGKR_base)                               |
| UniHGKR-base-beir     | adapted for evaluation on BEIR                                                                                             | [UniHGKR-base-beir](https://huggingface.co/ZhishanQ/UniHGKR-base-beir)         | [code for evaluation_beir](https://github.com/ZhishanQ/UniHGKR/tree/main/evaluation_beir)                             | 
| UniHGKR-7B            | LLM-based retriever                                                           | [UniHGKR-7B](https://huggingface.co/ZhishanQ/UniHGKR-7B)                |                                 [demo code to use](https://github.com/ZhishanQ/UniHGKR/tree/main/code_for_UniHGKR_7B) |
| UniHGKR-7B-pretrained | The model was trained after Stages 1 and 2. It needs to be fine-tuned before being used for an information retrieval task. | [UniHGKR-7B-pretrained](https://huggingface.co/ZhishanQ/UniHGKR-7B-pretrained) |                                                                                                                       |


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