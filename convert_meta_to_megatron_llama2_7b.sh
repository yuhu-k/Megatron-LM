export PYTHONPATH="$PYTHONPATH:$(pwd)"
LLAMA_META_FORMAT_DIR="llama-2-7b"
MEGATRON_FORMAT_DIR="llama-2-7b-me"
TOKENIZER_MODEL="/tmp2/Megatron-LM/tokenizer.model"
TP=1
PP=1

python tools/checkpoint/convert.py --model-type GPT \
   --loader llama_mistral \
   --saver megatron \
   --checkpoint-type meta \
   --model-size llama2-7B \
   --load-dir $LLAMA_META_FORMAT_DIR \
   --save-dir ${MEGATRON_FORMAT_DIR} \
   --tokenizer-model ${TOKENIZER_MODEL} \
   --target-tensor-parallel-size ${TP} \
   --target-pipeline-parallel-size ${PP}