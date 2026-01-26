## Purpose of Fork

This fork adapts DocBench to evaluate both local and cloud-based LLMs, with fixes for output alignment and reproducible evaluation.
Modifications to the source code allow for custom API base URLs, ensuring compatibility with non-OpenAI endpoints.

   


# DocBench: A Benchmark for Evaluating LLM-based Document Reading Systems
Paper Link: _[DocBench: A Benchmark for Evaluating LLM-based Document Reading Systems](https://arxiv.org/pdf/2407.10701)_

## Introduction

**DocBench** is a benchmark that takes raw PDF files and accompanying questions as inputs, with the objective of generating corresponding textual answers. It includes 229 real documents and 1,102 questions, spanning across five different domains and four major types of questions.

The construction pipeline consists of three pahses: (a) Document Collection; (b) QA-pair Generation; (c) Quality Check.

![](figs/intro.png)



## Dataset Overview

![](figs/dataset.png)

## Data

Data can be downloaded from: https://drive.google.com/drive/folders/1yxhF1lFF2gKeTNc8Wh0EyBdMT3M4pDYr?usp=sharing

## Implementations

We need keys from Hugging Face and OpenAI. (get your own keys and set them to the environment variables `HF_KEY` and `OPENAI_API_KEY`). Additionally, you need to set an environment variable called
'CLIENT_API_KEY' and BASE_URL. 

Things to be aware of:
1. If you are running models via OpenAI, then your OpenAI API key and client API key will be the same.
2. If running models locally, BASE_URL will be http://localhost:11434/v1, and a dummy value can be passed for CLIENT_API_KEY (e.g., 'ollama' if running model via Ollama).


### a. Download

Download the models to evaluate: 

```
bash download.sh
```

- ```YOUR_OWN_DIR```: where to save the downloaded models
- ```MODEL_TO_DOWNLOAD```: model name from hugging face

Alternative options include 

### b. Run

First, we create a remote Python environment:

```
python -m venv docbench-env’
```

Second, we run the models for inference:

```
python run.py \             
  --system gpt4_pl \
  --api_base <API base URL (for local models use: http://localhost:11434/v1)> \
  --model_name <model name> \
  --initial_folder 0

```

### c. Evaluate

Evaluate the results:

```bash
python evaluate.py \
  --system gpt4_pl \
  --resume_id 0
```

Notice: there could be some warnings for unexpected outputs. We could check the outputs according to the warning hint.


## Citation
If you find this work useful, please kindly cite our paper:
```
@misc{zou2024docbenchbenchmarkevaluatingllmbased,
      title={DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems}, 
      author={Anni Zou and Wenhao Yu and Hongming Zhang and Kaixin Ma and Deng Cai and Zhuosheng Zhang and Hai Zhao and Dong Yu},
      year={2024},
      eprint={2407.10701},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10701}, 
}
```
