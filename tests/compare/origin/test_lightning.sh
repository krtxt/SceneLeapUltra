# CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
#     +"checkpoint_path='experiments/h0.005_t10_r10_q0.5/checkpoints/epoch=223-val_loss=11.70.ckpt'" \


# Test single checkpoint
# CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
#     +"checkpoint_path='experiments/h0.005_t10_r10_q0.5/checkpoints/epoch=317-val_loss=11.29.ckpt'" \
#     data.test.batch_size=64 \
#     +force_retest=true 

# CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
#     +"checkpoint_path='experiments/t10_r10_q0.5/checkpoints/epoch=283-val_loss=11.44.ckpt'" \
#     data.test.batch_size=64 \
#     +force_retest=true 

# CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
#     +"checkpoint_path='experiments/h0.005_t10_r10_q0.5/checkpoints/epoch=317-val_loss=11.29.ckpt'" \
#     data.test.batch_size=64 

# Test all checkpoints in directory 
CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
    +train_root='experiments/test_528_1' \
    data.test.batch_size=34 \
#     +force_retest=true 

# CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
#     +train_root='experiments/t5_r5_q1' \
#     data.test.batch_size=64 \
#     +force_retest=true 

# CUDA_VISIBLE_DEVICES=2 python test_lightning.py \
#     +train_root='experiments/t10_r10_q0.5' \
#     data.test.batch_size=64 \
#     +force_retest=true 
    
# Force retest a checkpoint
# CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
#     +"checkpoint_path='experiments/h0.005_t10_r10_q0.5/checkpoints/epoch=223-val_loss=11.70.ckpt'" \
#     +force_retest=true \
