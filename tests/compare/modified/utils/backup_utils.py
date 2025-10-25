import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def backup_code(save_root: Path) -> None:
    """Back up a snapshot of current main codes with versioning"""
    backup_dir = save_root / "backups"
    
    existing_backups = [d for d in backup_dir.glob('v*') if d.is_dir()]
    next_version = 1
    if existing_backups:
        versions = [int(d.name[1:]) for d in existing_backups]
        next_version = max(versions) + 1
    
    version_dir = backup_dir / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    backup_list = [
        "datasets",
        "loss",
        "models",
        "tools", 
        "utils",
        "vis",
        "train_lightning.py",
        "train.py",
        "test_lightning.py",
        "test.py",
        "train_distributed.py",
        "*.sh",
        "config"
    ]
    
    root_dir = Path(__file__).parent.parent
    
    for item in backup_list:
        src = root_dir / item
        if src.is_dir():
            dst = version_dir / item
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
            logger.info(f"Backed up directory to v{next_version}: {item}")
        elif src.is_file():
            shutil.copy2(src, version_dir)
            logger.info(f"Backed up file to v{next_version}: {item}")
        elif "*" in item:
            for f in root_dir.glob(item):
                if f.is_file():
                    shutil.copy2(f, version_dir)
                    logger.info(f"Backed up file to v{next_version}: {f.name}") 