from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_llm(model_name_or_path, lora_name_or_path, save_path, cache_dir: str = None, token: str = None):
    
    print(f"model_name_or_path: {model_name_or_path}")
    print(f"lora_name_or_path: {lora_name_or_path}")
    print(f"save_path: {save_path}")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 cache_dir=cache_dir,
                                                 token=token,
                                                 trust_remote_code=True)
    
      # Check initial model data type
    initial_dtype = next(model.parameters()).dtype
    print(f"Initial model data type: {initial_dtype}")

    model = PeftModel.from_pretrained(model, lora_name_or_path)
    model = model.merge_and_unload()
    # model.half() 
    # Check merged model data type
    merged_dtype = next(model.parameters()).dtype
    
    print(f"Merged model data type: {merged_dtype}")

    model.save_pretrained(save_path)
    
    

    try:
        print("tokenizer from lora_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(lora_name_or_path)
    except:
        print("tokenizer from model_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  cache_dir=cache_dir,
                                                  token=token,
                                                  trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        if 'llara' in model_name_or_path.lower():
            tokenizer.padding_side = 'left'
            
    tokenizer.save_pretrained(save_path)
    print("done.")

    
model_name = 'UniHGKR-7B'

merge_llm('./pretrain/UniHGKR-7B-pretrained', './finetune/' + model_name, model_name + '-full')
