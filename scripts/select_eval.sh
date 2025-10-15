#!/bin/bash

implementations=("only" "semi")

for impl in "${implementations[@]}"; do
    python main.py --operator select --impl $impl --start 100 --end 130 --out_dir "./result/select/"
    python main.py --operator select --impl $impl --start 100 --end 130 --out_dir "./result/select/" --thinking 

    for example_num in "0" "3"; do
        python main.py --operator select --impl $impl --start 200 --end 215 --out_dir "./result/select/" --example_num $example_num
        python main.py --operator select --impl $impl --start 200 --end 215 --out_dir "./result/select/" --example_num $example_num --thinking

        python main.py --operator select --impl $impl --start 300 --end 315 --out_dir "./result/select/" --example_num $example_num
        python main.py --operator select --impl $impl --start 300 --end 315 --out_dir "./result/select/" --example_num $example_num --thinking
    done
done
