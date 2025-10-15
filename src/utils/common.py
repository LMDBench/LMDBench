import pandas as pd 
import numpy as np

def append_result(csv_path, pipeline, results):
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write(f"{pipeline},")
        for result in results:
            f.write(f"{result},")
        f.write(f"\n")
        f.flush()
    print(f"pipeline_{pipeline}: {results}")


def args_to_filename(args, suffix=".txt"):
    parts = [
        args.operator.value,
        args.impl.value,
        str(args.start),
        str(args.end)
    ]

    if hasattr(args, 'example_num') and args.example_num is not None:
        parts.append(str(args.example_num)+"shot")

    if hasattr(args, 'sort_algo') and args.sort_algo is not None:       
        parts.append(str(args.sort_algo))

    if args.thinking:
        parts.append('thinking')

    filename = "_".join(parts) + suffix
    return filename