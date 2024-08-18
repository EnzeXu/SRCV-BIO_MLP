#!/bin/bash
for seed in {0..4}
do
    for model in "MLP_with_grid"
    do
        for norm in 0
        do
            for gpu_id in 0
            do
                timestring=$(python shell_timestring.py)
                output_path="output/log/${model}_${timestring}_norm_${norm}.txt"
                echo "python -u main.py --gpu_id ${gpu_id} --model ${model} --data_norm ${norm} --seed ${seed} --epoch 1000 --timestring ${timestring} >> ${output_path}"
                python -u main.py --gpu_id ${gpu_id} --model ${model} --data_norm ${norm} --seed ${seed} --epoch 1000 --timestring ${timestring} >> ${output_path}
            done
        done
    done
done
