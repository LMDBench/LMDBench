#!/bin/bash

python main.py --operator impute --impl only --start 100 --end 130 --out_dir "./result/impute/"
python main.py --operator impute --impl only --start 100 --end 130 --out_dir "./result/impute/" --thinking
python main.py --operator impute --impl semi --start 100 --end 130 --out_dir "./result/impute/" --example_num 0
python main.py --operator impute --impl semi --start 100 --end 130 --out_dir "./result/impute/" --example_num 0 --thinking
python main.py --operator impute --impl semi --start 100 --end 130 --out_dir "./result/impute/" --example_num 3
python main.py --operator impute --impl semi --start 100 --end 130 --out_dir "./result/impute/" --example_num 3 --thinking


python main.py --operator impute --impl only --start 200 --end 230 --out_dir "./result/impute/"
python main.py --operator impute --impl only --start 200 --end 230 --out_dir "./result/impute/" --thinking
python main.py --operator impute --impl semi --start 200 --end 230 --out_dir "./result/impute/"
python main.py --operator impute --impl semi --start 200 --end 230 --out_dir "./result/impute/" --thinking


python main.py --operator impute --impl only --start 300 --end 315 --out_dir "./result/impute/" --example_num 0
python main.py --operator impute --impl only --start 300 --end 315 --out_dir "./result/impute/" --example_num 3
python main.py --operator impute --impl only --start 300 --end 315 --out_dir "./result/impute/" --example_num 0 --thinking
python main.py --operator impute --impl only --start 300 --end 315 --out_dir "./result/impute/" --example_num 3 --thinking