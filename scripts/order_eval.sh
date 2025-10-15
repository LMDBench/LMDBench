#!/bin/bash

python main.py --operator order --impl only --end 30 --out_dir "./result/order/"
python main.py --operator order --impl semi --end 30 --sort_algo simple --out_dir "./result/order/"
python main.py --operator order --impl semi --end 30 --sort_algo heap --out_dir "./result/order/"
python main.py --operator order --impl semi_prime --end 30 --out_dir "./result/order/" 

python main.py --operator order --impl only --end 30 --out_dir "./result/order/" --thinking
python main.py --operator order --impl semi --end 30 --sort_algo simple --out_dir "./result/order/" --thinking
python main.py --operator order --impl semi --end 30 --sort_algo heap --out_dir "./result/order/" --thinking
python main.py --operator order --impl semi_prime --end 30 --out_dir "./result/order/" --thinking

