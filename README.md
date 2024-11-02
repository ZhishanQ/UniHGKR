<h2 align="center"> <a href="https://arxiv.org/abs/2410.20163">UniHGKR: Unified Instruction-aware Heterogeneous Knowledge Retrievers</a></h2>

🌟 This is the official repository for Dense Heterogeneous Knowledge Retrievers: **UniHGKR**, and the heterogeneous knowledge retrieval benchmark **CompMix-IR**.

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2410.20163-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.20163)

</h5>

## Abstract

Existing information retrieval (IR) models often assume a homogeneous structure for knowledge sources and user queries, limiting their applicability in real-world settings where retrieval is inherently heterogeneous and diverse.
In this paper, we introduce UniHGKR, a unified instruction-aware heterogeneous knowledge retriever that (1) builds a unified retrieval space for heterogeneous knowledge and (2) follows diverse user instructions to retrieve knowledge of specified types. 
UniHGKR consists of three principal stages: heterogeneous self-supervised pretraining, text-anchored embedding alignment, and instruction-aware retriever fine-tuning, enabling it to generalize across varied retrieval contexts. This framework is highly scalable, with a BERT-based version and a UniHGKR-7B version trained on large language models. 
Also, we introduce CompMix-IR, the first native heterogeneous knowledge retrieval benchmark. It includes two retrieval scenarios with various instructions, over 9,400 question-answer (QA) pairs, and a corpus of 10 million entries, covering four different types of data.
Extensive experiments show that UniHGKR consistently outperforms state-of-the-art methods on CompMix-IR, achieving up to 6.36% and 54.23% relative improvements in two scenarios, respectively.
Finally, by equipping our retriever for open-domain heterogeneous QA systems, we achieve a new state-of-the-art result on the popular [ConvMix](https://convinse.mpi-inf.mpg.de/) task, with an absolute improvement of up to 4.80 points.


## Notes:

**We are preparing to update more code and benchmark datasets. Please be patient.**

## 1. CompMix-IR Benchmark

For more detailed information about the CompMix-IR Benchmark, please refer to the [CompMix_IR](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR) directory.

### 1.1 Corpus of CompMix-IR:

Download from 🤗 HuggingFace Dataset: [Link](https://huggingface.co/datasets/ZhishanQ/CompMix-IR) or ☁️ Google Drive: [Link](https://drive.google.com/file/d/1sDmPieBkAnO9Rb7oDDXAgRDd5SRo_rPP/view?usp=sharing) .

The complete version of the CompMix_IR heterogeneous knowledge corpus is approximately 3-4 GB in size. We also provide a smaller file, which is a subset, to help readers understand its content and structure: [subset of corpus](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/subset_kb_wikipedia_mixed_rd.json)


### 1.2 QA pairs of CompMix:

CompMix QA pairs: [CompMix](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/CompMix)

ConvMix QA pairs: [ConvMix_annotated](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/ConvMix_annotated)

or Huggingface dataset:

[CompMix](https://huggingface.co/datasets/pchristm/CompMix), [ConvMix](https://huggingface.co/datasets/pchristm/ConvMix)

### 1.3 Code to evaluate 

Code to evaluate whether the retrieved evidence is positive to the question:

[Code to judge relevance](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/eval_part)

### 1.4 Data-Text Pairs 

It is used in training stages 1 and 2.

Download from 🤗 HuggingFace Dataset: [Link](https://huggingface.co/datasets/ZhishanQ/UniHGKR_Date_Text_Pairs/tree/main) or ☁️ Google Drive: [Link](https://drive.google.com/file/d/1AOyY0T_FQo7Br6o7KfkNSnoW9L9dZYXb/view?usp=sharing) .

The complete version of Data-Text Pairs is about 1.2 GB. We also provide a smaller file, which is a subset, to help readers understand its content and structure: [subset of data-text pairs](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR/data_2_text_subset.json)

The [CompMix_IR](https://github.com/ZhishanQ/UniHGKR/tree/main/CompMix_IR) directory provides detailed explanations for the keys within each dict item.

## 2. UniHGKR model checkpoints
 
| Mdeol Name            | Description                                                                                                                | 🤗 Huggingface  Link                                                              | Usage Example                                                                                                         |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| UniHGKR-base          | adapted for evaluation on CompMix-IR                                                                                       | [UniHGKR-base](https://huggingface.co/ZhishanQ/UniHGKR-base)                   | [demo code to use](https://github.com/ZhishanQ/UniHGKR/tree/main/code_for_UniHGKR_base)                               |
| UniHGKR-base-beir     | adapted for evaluation on BEIR                                                                                             | [UniHGKR-base-beir](https://huggingface.co/ZhishanQ/UniHGKR-base-beir)         | [code for evaluation_beir](https://github.com/ZhishanQ/UniHGKR/tree/main/evaluation_beir)                             | 
| UniHGKR-7B            | LLM-based retriever                                                           | [UniHGKR-7B](https://huggingface.co/ZhishanQ/UniHGKR-7B)                |                                 [demo code to use](https://github.com/ZhishanQ/UniHGKR/tree/main/code_for_UniHGKR_7B) |
| UniHGKR-7B-pretrained | The model was trained after Stages 1 and 2. It needs to be fine-tuned before being used for an information retrieval task. | [UniHGKR-7B-pretrained](https://huggingface.co/ZhishanQ/UniHGKR-7B-pretrained) |                                                                                                                       |


## 3. Code to train and evalutation

#### 3.1 Evalutation on CompMix-IR

#### 3.2 Evalutation on Convmix

#### 3.3 Evalutation on BERI

Our variant model **UniHGKR-base-beir** adapted for evaluation on BEIR can be found at: https://huggingface.co/ZhishanQ/UniHGKR-base-beir

The code for evaluation on BEIR at: [evaluation_beir](https://github.com/ZhishanQ/UniHGKR/tree/main/evaluation_beir).


## ✏️ Citation
If you find our paper and resource useful in your research, please consider giving a star :star: and citation :pencil:.

```
@article{min2024unihgkr,
  title={UniHGKR: Unified Instruction-aware Heterogeneous Knowledge Retrievers},
  author={Min, Dehai and Xu, Zhiyang and Qi, Guilin and Huang, Lifu and You, Chenyu},
  journal={arXiv preprint arXiv:2410.20163},
  year={2024}
}
```

## 📧 contact
```

dmin0007[at]student.monash.edu

```

