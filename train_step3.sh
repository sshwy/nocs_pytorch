python final.py \
    --config-file configs/NOCSrcnn.yaml \
    --resume \
    OUTPUT_DIR output_step3 \
    SOLVER.BASE_LR 0.00001 \
    MODEL.BACKBONE.FREEZE_AT 2 \
    SOLVER.MAX_ITER 400000 \
    SOLVER.STEPS "(300000, 330000)" \
    MODEL.WEIGHTS output_step2/model_final.pth

# SOLVER.STEPS The iteration number to decrease learning rate by GAMMA.