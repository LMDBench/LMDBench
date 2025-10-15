
import argparse
from src.core.enums import ImplType, OperandType, OperatorType
from src.utils.common import append_result, args_to_filename
from eval import *
from multiprocessing import Process, Manager


def wrap_query(id, return_dict, args):
    query_name = f"pipeline_{id}"
    query = globals().get(query_name, None)
    if query is None:
        return_dict['result'] = 'Not Implemented'
        print(f"pipeline_{id}" + " is not implemented")
    else:
        if args.operator.value!="e2e":
            return_dict['result'], _ = query(args) 
        else:
            return_dict['result'] = query()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True, type=str, choices=[e.value for e in OperatorType])
    parser.add_argument("--impl", required=True, type=str,choices=[e.value for e in ImplType])
    parser.add_argument("--thinking",action="store_true", default=False)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="./result/")
    parser.add_argument("--example_num", type=int, default=None)
    parser.add_argument("--sort_algo", type=str, default=None)


    args = parser.parse_args()
    args.operator = OperatorType(args.operator)
    args.impl = ImplType(args.impl) 
    if args.operator.value != "e2e":
        print("=" * 120)
        args_string = f"Operator: {args.operator}, Impl: {args.impl}, Start: {args.start}, End: {args.end}, Thinking: {args.thinking}"
        if args.example_num is not None:
            args_string += f", Example: {args.example_num}"
        if args.sort_algo is not None:
            args_string += f", Sort_Algo: {args.sort_algo}"
        print(args_string)
        print("=" * 120)
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    for id in range(args.start, args.end):
        id = args.operator.value + str(id) if args.operator.value != "e2e" else str(id)
        print(f"pipeline_{id}" + " starts")
        manager = Manager()
        return_dict = manager.dict()
        try:
            p = Process(target=wrap_query, args=(id, return_dict, args))
            p.start()
            p.join(timeout=30*60)
            if p.is_alive():
                p.terminate()
                results = ['TIMEOUT']
            else:
                results = return_dict.get('result', ['ERROR'])
                print(results)
                if isinstance(results, tuple):
                    results = list(results)
                else:
                    results = [results]
        except Exception as e:
            print(e)
            results = ['ERROR']

        if args.operator.value == "e2e":
            file_name = "e2e_results.csv"
        else:
            file_name = args_to_filename(args, ".csv")
        from os.path import join
        append_result(join(args.out_dir, file_name), id, results)



if __name__ == "__main__":
    main()

