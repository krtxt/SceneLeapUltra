import logging
from typing import Dict
from .log_colors import HEADER, BLUE, GREEN, ENDC
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
                           val_detailed_loss: Dict[str, float]) -> None:
    logging.info(f'{GREEN}=== Epoch {epoch} - Validation Results ==={ENDC}')
    logging.info(f'{BLUE}Total validation batches: {num_batches}{ENDC}')

    avg_loss_str = f"{avg_loss:.4f}"
    emoji_loss = convert_number_to_emoji(avg_loss_str)
    logging.info(f'{BLUE}{"Average Loss:":<21s} {avg_loss_str} {emoji_loss}{ENDC}')
    logging.info(f'{BLUE}{"Loss Std:":<21s} {loss_std:.4f}{ENDC}')
    logging.info(f'{BLUE}{"Loss Min:":<21s} {loss_min:.4f}{ENDC}')
    logging.info(f'{BLUE}{"Loss Max:":<21s} {loss_max:.4f}{ENDC}')

    logging.info(f'{BLUE}--- Detailed Loss Breakdown ---{ENDC}')
    for k, v in val_detailed_loss.items():
        v_str = f"{v:.4f}"
        emoji_v = convert_number_to_emoji(v_str)
        logging.info(f'{BLUE}{k.title() + ":":<21s} {v_str} {emoji_v}{ENDC}')

    logging.info(f'{GREEN}=== Validation Summary Complete ==={ENDC}') 