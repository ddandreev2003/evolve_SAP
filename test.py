from pathlib import Path
import torch
from SAP_pipeline_flux import SapFlux

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
LOCAL_DIR = Path.home() / ".cache" / "sap_flux_models" / "FLUX.2-klein-base-4B"

# 1) Скачиваем из HF и сохраняем локально
model = SapFlux.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(LOCAL_DIR))
print(f"Saved locally: {LOCAL_DIR}")

# 2) Проверка: грузим уже только локально (без сети)
model_local = SapFlux.from_pretrained(
    str(LOCAL_DIR),
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
print("Loaded from local only")
