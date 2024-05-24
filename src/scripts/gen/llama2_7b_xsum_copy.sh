model=llama2
modelsize=0 
dataset=xsum
shots=1
device=9

# GRIFFIN
python eval_gen_copy.py \
    --dataset $dataset \
    --shots $shots \
    --model_arch $model \
    --model_size $modelsize \
    --density 0.5 \
    --selection_method topk \
    --device cuda:$device \
    --sample_num 10 \
    --max_token 64

# python eval_gen.py \
#     --dataset $dataset \
#     --shots $shots \
#     --model_arch $model \
#     --model_size $modelsize \
#     --density 0.5 \
#     --selection_method topk \
#     --device cuda:$device \
#     --sample_num 10 \
#     --max_token 64