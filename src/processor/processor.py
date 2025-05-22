from src.models import VLChatProcessor

def get_processor(model_name):
    processor = VLChatProcessor.from_pretrained(model_name)
    processor.tokenizer.add_tokens(['<|box_start|>', '<|box_end|>', '<|obj_start|>', '<|obj_end|>'])
    return processor