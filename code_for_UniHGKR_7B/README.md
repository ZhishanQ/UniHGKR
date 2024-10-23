
## UniHGKR-7B



 
| Mdeol Name            | Description                                                                                                                | Huggingface  Link                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| UniHGKR-7B            | LLM-based retriever                                                           | [UniHGKR-7B](https://huggingface.co/ZhishanQ/UniHGKR-7B)                                                                 |
| UniHGKR-7B-pretrained | The model was trained after Stages 1 and 2. It needs to be fine-tuned before being used for an information retrieval task. | [UniHGKR-7B-pretrained](https://huggingface.co/ZhishanQ/UniHGKR-7B-pretrained) |



Instructions for our retrievers

```shell
general_ins = "Given a question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst = "Given a question, retrieve relevant evidence that can answer the question from {} sources: "

general_ins_with_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from all knowledge sources: "
single_source_inst_domain = "Given a {} domain question, retrieve relevant evidence that can answer the question from {} sources: "
```

You can prepend the instructions provided above to the input query to achieve optimal performance.

Our model is trained based on [LLARA-pretrained](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/LLARA), allowing seamless integration of its training and usage code.

We will soon release the evaluation code for UniHGKR-7B on CompMix-IR.