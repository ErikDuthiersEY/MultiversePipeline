from pathlib import Path
from dotenv import load_dotenv
import os

from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

load_dotenv()

# === CONFIGURACION ===
ROOT_DIR = Path.cwd()
PARQUET_DIR = ROOT_DIR / "datasets" / "processed"

SUBSCRIPTION_ID = os.getenv("AZ_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZ_RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("AZ_ML_WORKSPACE")
MODE = "files"                            
VERSION = "1"                          # Versión base; se incrementará si existe
ASSET_PREFIX = "eval_"                     
ASSET_NAME = "evaluation_parquet_bundle"    
DATASTORE = None                           

# Verifica vars de Azure
if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME]):
    raise ValueError("Faltan variables en .env: AZ_SUBSCRIPTION_ID, AZ_RESOURCE_GROUP o AZ_ML_WORKSPACE.")

# Verifica carpeta
assert PARQUET_DIR.exists(), f"No existe {PARQUET_DIR}"
files = list(PARQUET_DIR.glob("*.parquet"))
print(f"Encontrados {len(files)} .parquet en {PARQUET_DIR}")

# Cliente Azure ML
cred = InteractiveBrowserCredential()
ml_client = MLClient(cred, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
print(f"Conectado a workspace: {ml_client.workspace_name}")

# Función helper: Encuentra versión única para un nombre de asset
def get_unique_version(ml_client, asset_name: str, base_version: str) -> str:
    version = int(base_version)
    while True:
        try:
            # Intenta listar versiones; si existe esta versión, incrementa
            existing = ml_client.data.get(name=asset_name, version=str(version))
            print(f"Asset {asset_name}:{version} ya existe. Intentando versión {version + 1}...")
            version += 1
        except Exception:
            # No existe, úsala
            return str(version)

# Función para registrar archivo (con versión única)
def register_file(ml_client, path: Path, name: str, base_version: str, datastore: str | None):
    unique_version = get_unique_version(ml_client, name, base_version)
    asset = Data(
        name=name,
        version=unique_version,
        type=AssetTypes.URI_FILE,
        path=str(path),
        datastore=datastore,
        description=f"Parquet file from {path.name}",
    )
    created = ml_client.data.create_or_update(asset)
    print(f"[FILE] {created.name}:{created.version} → {created.path}")

# Función para registrar carpeta (con versión única)
def register_folder(ml_client, folder: Path, name: str, base_version: str, datastore: str | None):
    unique_version = get_unique_version(ml_client, name, base_version)
    asset = Data(
        name=name,
        version=unique_version,
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