from dpr import core
from dpr import utils
utils.create_dirs()
tokenizer, tokenizer_ans, model, model_ans, pipe, whisper_model = core.load_models()
