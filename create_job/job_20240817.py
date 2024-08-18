template = """#!/bin/bash
for seed in {{0..4}}
do
    for model in "{0}"
    do
        for norm in {1}
        do
            for gpu_id in {2}
            do
                timestring=$(python shell_timestring.py)
                output_path="output/log/${{model}}_${{timestring}}_norm_${{norm}}.txt"
                echo "python -u main.py --gpu_id ${{gpu_id}} --model ${{model}} --data_norm ${{norm}} --seed ${{seed}} --epoch 1000 >> ${{output_path}}"
                python -u main.py --gpu_id ${{gpu_id}} --model ${{model}} --data_norm ${{norm}} --seed ${{seed}} --epoch 1000 >> ${{output_path}}
            done
        done
    done
done
"""

model_list = ["FNO", "MLP", "MLP_with_grid"]
norm_list = ["0", "1"]
gpu_dic = {
    "0": 0,
    "1": 2,
}
for i in model_list:
    for j in norm_list:
        with open(f"jobs/job_20240817_{i}_{j}.sh", "w") as f:
            print(i, j, gpu_dic[j])
            f.write(template.format(i, j, gpu_dic[j]))
