import sys
import torch
import validate as v

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
processor, model = v.load_model_stack(v.MODEL_DIR, device, dtype)

for path in sys.argv[2:]:
    raw, prediction, exact, latency_ms = v.infer_single(processor, model, path, device)
    print(path.split("/")[-1], "->", prediction, "| raw:", raw, "|", int(latency_ms), "ms")
