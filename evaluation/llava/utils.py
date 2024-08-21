def get_prefix_suffix(model):
    if "1.5" in model:
        prefix = '<image>\nUSER: '
        suffix = '\nASSISTANT:'
    elif "1.6" in model and "mistral" in model:
        prefix = '[INST] <image>\n'
        suffix = ' [/INST]'
    elif "1.6" in model and "vicuna" in model:
        prefix = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and precise answers to the human's questions. USER: <image>\n"
        suffix = "\nASSISTANT:"
    elif "llava-v1.6-34b-hf" in model:
        prefix = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n"
        suffix = "<|im_end|><|im_start|>assistant\n"
    return prefix, suffix