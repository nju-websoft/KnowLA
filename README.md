# KnowLA
KnowLA: Enhancing Parameter-efficient Finetuning with Knowledgeable Adaptation, NAACL 2024

------

## Preparation of KnowLA

1. Install the updated `loralib`:

```
pip install -r requirements.txt
```

2. Download the Llama2 checkpoints and modified modeling_llama.py from [here](https://huggingface.co/luoxindi/llama2_knowla/tree/main). 
3. Download the LoRA checkpoints of [KnowLA](https://huggingface.co/luoxindi/llama2-lora-32/tree/main). ,  [LLama2-lora (*r=16*)](https://huggingface.co/luoxindi/llama2-lora-r16/tree/main) and [LLama2-lora (*r=32*)](https://huggingface.co/luoxindi/llama2-lora-32/tree/main). 
4. Download the datasets from [here](https://huggingface.co/luoxindi/data/tree/main)
5. Download the KG configurations  from [here](https://huggingface.co/luoxindi/kgs/tree/main). And move them under the `data` folder.

## Quickstart

1. Train Llama2-lora:

```bash
python finetune.py \
    --base_model='/llama2_7B' \
    --num_epochs=3 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./llama2-lora' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=4
```

2. Train KnowLA:

```bash
python finetune_kg.py \
    --base_model='/llama2_7B' \
    --num_epochs=3 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./llama2-lora-cn' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=4
```

We recommend you to change the hyperparameter `layer_insertion` in the file `llama2-7B/config.json` to dynamically allocate the KG adapters to the corresponding layer, like `layer_insertion: [16, 32]`.

3. Test csqa/siqa for KnowLA:

```bash
python test_csqa.py \
    --base_model='/llama2_7B' \
    --is_KG=True \
    --lora_weights="./llama2-lora-cn" \
    --dataset="siqa"
```

4. Test other tasks for KnowLA:

```bash
python test.py \
    --base_model='/llama2_7B' \
    --is_KG=True \
    --lora_weights="./llama2-lora-cn" \
```

 test.py can be replaced by any test file such as test_wqsp.py.

## Notice

We finished these training and testing processes with an A800. And we offload the KG pretrained embeddings to the CPU to reduce VRAM allocation. During training, the VRAM usage and training time of KnowLA and Llama2 are similar.

We find that when we reduce the trainable parameters in KnowLA, the performance of the LLM is improved. Therefore, we recommend you to change the hyperparameter `kg_intermediate_size`  from 1024 to 100 in the `llama2-7B/config.json` for training. This weight can be downloaded from [here](). 

For loading the modified `modeling_llama.py` file, we use `shutil.copyfile` to write it in the conda environment. If you do not want to load in this way, consider using the following code to load the modified `modeling_llama.py` file.

```python
model = AutoModel.from_pretrained(
        base_model,
        config = config,
        device_map="auto",
        trust_remote_code=True,
    )
```

