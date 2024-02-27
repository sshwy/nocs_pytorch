python final.py \
    --config-file configs/NOCSrcnn.yaml \
    --resume \
    OUTPUT_DIR output_step2 \
    SOLVER.BASE_LR 0.0001 \
    MODEL.BACKBONE.FREEZE_AT 4 \
    SOLVER.MAX_ITER 130000 \
    SOLVER.STEPS "(105000, 115000)" \
    MODEL.WEIGHTS output_step1/model_final.pth

# SOLVER.STEPS The iteration number to decrease learning rate by GAMMA.