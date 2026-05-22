"""PYTHONSTARTUP: wrap OpenEvolve worker init (GPU assign happens inside wrapper)."""
from openevolve_sap.core.gpu_worker import patch_openevolve_worker_init

patch_openevolve_worker_init()
