CASES = {
    "debug_fp32": {
        "rows": 128,
        "hidden": 256,
        "dtype": "float32",
        "eps": 1e-5,
        "warmup": 10,
        "iters": 50,
    },
    "main_fp32": {
        "rows": 512,
        "hidden": 768,
        "dtype": "float32",
        "eps": 1e-5,
        "warmup": 20,
        "iters": 100,
    },
    "main_fp16": {
        "rows": 512,
        "hidden": 768,
        "dtype": "float16",
        "eps": 1e-5,
        "warmup": 20,
        "iters": 100,
    },
    "large_fp16": {
        "rows": 4096,
        "hidden": 4096,
        "dtype": "float16",
        "eps": 1e-5,
        "warmup": 30,
        "iters": 200,
    },
}

