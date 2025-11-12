# -*- coding: utf-8 -*-
"""
Ejecuta esto después de generar los .parquet en OUT_DIR.
Asegúrate de tener:
- pip install azure-ai-ml azure-identity
- az login (o AZURE_CLIENT_ID/TENANT_ID/CLIENT_SECRET en env vars)

Modos:
- files: Un asset por .parquet (recomendado para individualidad).
- folder: Un asset para toda la carpeta.

Adapta los params de Azure abajo.
"""

from pathlib import Path
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# === CONFIGURACIÓN ===
PARQUET_DIR = Path(r"C:\Users\WJ724NE\OneDrive - EY\Documents\MultiversePipeline\datasets\processed")
SUBSCRIPTION_ID = "3bcd6b3d-11a3-4a3e-9e9e-7beb46101ac4" 
RESOURCE_GROUP = "ws-multi-studio"   
WORKSPACE_NAME = "multistudioaml"             
MODE = "files"                            
VERSION = "1"                           
ASSET_PREFIX = "eval_"                     
ASSET_NAME = "evaluation_parquet_bundle"    
DATASTORE = None                           

# Verifica carpeta
assert PARQUET_DIR.exists(), f"No existe {PARQUET_DIR}"
files = list(PARQUET_DIR.glob("*.parquet"))
print(f"Encontrados {len(files)} .parquet en {PARQUET_DIR}")

# Cliente Azure ML
cred = InteractiveBrowserCredential()
ml_client = MLClient(cred, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
print(f"Conectado a workspace: {ml_client.workspace_name}")

# Función para registrar archivo
def register_file(ml_client, path: Path, name: str, version: str, datastore: str | None):
    asset = Data(
        name=name,
        version=version,
        type=AssetTypes.URI_FILE,
        path=str(path),
        datastore=datastore,
        description=f"Parquet file from {path.name}",
    )
    created = ml_client.data.create_or_update(asset)
    print(f"[FILE] {created.name}:{created.version} → {created.path}")

# Función para registrar carpeta
def register_folder(ml_client, folder: Path, name: str, version: str, datastore: str | None):
    asset = Data(
        name=name,
        version=version,
        type=AssetTypes.URI_FOLDER,
        path=str(folder),
        datastore=datastore,
        description=f"Folder with parquet files from {folder}",
    )
    created = ml_client.data.create_or_update(asset)
    print(f"[FOLDER] {created.name}:{created.version} → {created.path}")

# Procesar
if MODE == "files":
    for f in sorted(files):
        stem = f.stem.strip().replace(" ", "_").lower()
        name = f"{ASSET_PREFIX}{stem}"
        register_file(ml_client, f, name, VERSION, DATASTORE)
else:
    register_folder(ml_client, PARQUET_DIR, ASSET_NAME, VERSION, DATASTORE)

print("Subida completa en Azure ML Studio → Data → Data assets.")