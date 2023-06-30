MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-sft,use_fast=False"
TASK="jcommonsenseqa-1.1-0.4,jnli-1.1-0.4,marc_ja-1.1-0.4,jsquad-1.1-0.4,xlsum_ja-1.0-0.4"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2,3,3,3,1" --device "cuda" --output_path "models/rinna-japanese-gpt-neox-3.6b-instruction-sft/result.json"