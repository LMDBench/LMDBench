#!/bin/bash

implementations=("only" "semi" "semi_prime")

for impl in "${implementations[@]}"; do
    python main.py --operator map --impl $impl --start 100 --end 130 --out_dir "./result/map/"
    python main.py --operator map --impl $impl --start 100 --end 130 --out_dir "./result/map/" --thinking

    python main.py --operator map --impl $impl --start 200 --end 215 --out_dir "./result/map/"
    python main.py --operator map --impl $impl --start 200 --end 215 --out_dir "./result/map/" --thinking

    for example_num in "0" "3"; do
        python main.py --operator map --impl $impl --start 300 --end 315 --out_dir "./result/map/" --example_num $example_num
        python main.py --operator map --impl $impl --start 300 --end 315 --out_dir "./result/map/" --example_num $example_num --thinking
    done
done
