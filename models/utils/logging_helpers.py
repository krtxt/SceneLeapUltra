import logging
from typing import Dict

from .log_colors import BLUE, ENDC, GREEN, HEADER

# from .text_utils import convert_number_to_emoji

number_emoji_map = {
    '0': '0️⃣ ',
    '1': '1️⃣ ',
    '2': '2️⃣ ',
    '3': '3️⃣ ',
    '4': '4️⃣ ',
    '5': '5️⃣ ',
    '6': '6️⃣ ',
    '7': '7️⃣ ',
    '8': '8️⃣ ',
    '9': '9️⃣ ',
    '.': '🔸'
}

def convert_number_to_emoji(number_str: str) -> str:
    return ''.join(number_emoji_map.get(ch, ch) for ch in number_str) 

def log_validation_summary(epoch: int,
                           num_batches: int,
                           avg_loss: float,
                           loss_std: float,
                           loss_min: float,
                           loss_max: float,
                           val_detailed_loss: Dict[str, float],
                           val_set_metrics: Dict[str, float] = None,
                           val_quality_metrics: Dict[str, float] = None) -> None:
    logging.info(f'{GREEN}=== Epoch {epoch} - Validation Results ==={ENDC}')
    logging.info(f'{BLUE}Total validation batches: {num_batches}{ENDC}')

    avg_loss_str = f"{avg_loss:.4f}"
    # emoji_loss = convert_number_to_emoji(avg_loss_str)  # 注释掉emoji的输出
    logging.info(f'{BLUE}{"Average Loss:":<21s} {avg_loss_str}{ENDC}')
    logging.info(f'{BLUE}{"Loss Std:":<21s} {loss_std:.4f}{ENDC}')
    logging.info(f'{BLUE}{"Loss Min:":<21s} {loss_min:.4f}{ENDC}')
    logging.info(f'{BLUE}{"Loss Max:":<21s} {loss_max:.4f}{ENDC}')

    logging.info(f'{BLUE}--- Detailed Loss Breakdown ---{ENDC}')
    for k, v in val_detailed_loss.items():
        v_str = f"{v:.4f}"
        # emoji_v = convert_number_to_emoji(v_str)  # 注释掉emoji的输出
        logging.info(f'{BLUE}{k.title() + ":":<21s} {v_str}{ENDC}')

    # Optional: Set-level metrics
    if isinstance(val_set_metrics, dict) and len(val_set_metrics) > 0:
        logging.info(f'{BLUE}--- Set Metrics ---{ENDC}')
        for k, v in val_set_metrics.items():
            v_str = f"{v:.4f}"
            # emoji_v = convert_number_to_emoji(v_str)  # 注释掉emoji的输出
            logging.info(f'{BLUE}{k + ":":<21s} {v_str}{ENDC}')

    # Optional: Quality metrics (e.g., penetration_penalty, contact_quality)
    if isinstance(val_quality_metrics, dict) and len(val_quality_metrics) > 0:
        logging.info(f'{BLUE}--- Quality Metrics ---{ENDC}')
        for k, v in val_quality_metrics.items():
            v_str = f"{v:.4f}"
            # emoji_v = convert_number_to_emoji(v_str)  # 注释掉emoji的输出
            logging.info(f"{BLUE}{k.replace('_', ' ').title() + ':':<21s} {v_str}{ENDC}")

    logging.info(f'{GREEN}=== Validation Summary Complete ==={ENDC}') 