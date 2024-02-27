python final.py \
    --config-file configs/NOCSrcnn.yaml \
    --resume \
    OUTPUT_DIR output_step1 \
    MODEL.BACKBONE.FREEZE_AT 6 \
    SOLVER.MAX_ITER 100000 \
    SOLVER.STEPS "(75000, 85000)"

# SOLVER.STEPS The iteration number to decrease learning rate by GAMMA.