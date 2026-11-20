# FirstBreakAI-Code

Code and notebooks for [FBA Labs](https://github.com/thefirehacker).

## How we will learn

This repo will grow into a practical companion for two very long courses by [Umar Jamil](https://www.youtube.com/@umarjamilai). The videos are the lectures. Here we will write the code and the exercises that make the same ideas easy to follow, then go deeper than the video alone.

Each topic gets:

- **Detailed code** you can run, broken into small steps.
- **Custom exercises** that are hands-on and written so the hard parts stay understandable.
- **Extra depth** where a video moves fast: derivations, small experiments, and notes on what the code is actually doing.
- **Short blogs** that follow the exercises, so you can read the idea again after you have coded it.

### The two courses

1. **[Flash Attention derived and coded from first principles with Triton (Python)](https://www.youtube.com/watch?v=zy8ChVd_oTM)** (~7.5 hours). Attention, safe and online softmax, GPU memory, block matrix multiplication, then the Flash Attention forward and backward passes written in Triton.
2. **[Building a distributed training framework from first principles](https://www.youtube.com/watch?v=XoGvCBRnwLs)** (~19 hours). A full training stack, built up piece by piece:
   - Model pieces: RoPE, YaRN, MLA, MoE
   - Weight initialization
   - Distributed training: pipeline (PP), HSDP, context (CP), tensor (TP), expert (EP), and expert-tensor (ETP) parallelism
   - Mixed precision
   - Data loaders, sequence packing, and gradient adjustments
   - Loss-free MoE load balancing
   - DTensor and collective operations

More posts will follow as each exercise lands. The BubblSpace notebook below is the first runnable piece already in the repo.

## Start here: BubblSpace MoE-0.9B-A0.2B

**Notebook:** [`notebooks/bubblspace_moe0_9b_a0_2b.ipynb`](notebooks/bubblspace_moe0_9b_a0_2b.ipynb)

This is the runnable Colab notebook for [bubblspace-MoE-0.9B-A0.2B](https://github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B) — a from-scratch Mixture-of-Experts language model (~915M total / ~211M active parameters). Open it in Colab with a GPU (T4 is enough) and run the cells top to bottom.

The notebook:

1. Clones the weights and model code from [TheFireHacker/bubblspace-moe0.9b-a0.2b](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b) and installs `requirements.txt`.
2. Loads the custom `Transformer` (this checkpoint is **not** a stock `transformers` model) from `model.safetensors` plus the BubblSpace tokenizer.
3. Generates text with temperature and top-k sampling, then opens a short interactive prompt loop.

It is a **base** checkpoint (FineWeb-Edu pretraining, no instruction tuning). It continues text. Short factual prompts work better than chat-style questions.

| | |
|---|---|
| Training code | [github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B](https://github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B) |
| Weights | [huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b) |
| Notebook | [`notebooks/bubblspace_moe0_9b_a0_2b.ipynb`](notebooks/bubblspace_moe0_9b_a0_2b.ipynb) |
| Shape | 16 layers, `d_model=1024`, 64 experts, top-8, context 2048 |
| License | Apache 2.0 |

### Run it locally

Same steps the notebook runs, if you already have a CUDA machine:

```bash
git lfs install
git clone https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b
cd bubblspace-moe0.9b-a0.2b
pip install -r requirements.txt
python generate.py --prompt "Albert Einstein what is his most notable work" --max_new_tokens 100
```

## Other notebooks

| Notebook | What it is |
|---|---|
| [`notebooks/bubblspace_moe0_9b_a0_2b.ipynb`](notebooks/bubblspace_moe0_9b_a0_2b.ipynb) | Load and generate with BubblSpace MoE-0.9B-A0.2B |
| [`Lesson01/HF_DataSet_Analysis.ipynb`](Lesson01/HF_DataSet_Analysis.ipynb) | Lesson 01 dataset inspection (Hugging Face datasets) |

## License

Apache License 2.0. See [LICENSE](LICENSE).
