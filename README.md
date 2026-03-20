```bash
 nohup python3 /sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model /mnt/data/models/DeepSeek-V3.2/ \
    --dtype fp8_w8a8 \
    --tp-size 8 \
    --ep-size 8 \
    --topk-ids-dir {topk_ids_dir} \
    --tune \
    --disable-shared-experts-fusion \
    > tune.out 2>&1 &
```