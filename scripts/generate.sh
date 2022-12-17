export CUDA_VISIBLE_DEVICES=3

python ../tools/run_net.py --config-file ../projects/neus/configs/dtu_trainable/neus_dtu24_womask_barf_two_optim_1.py --type self_calibrated_barf --task generate_depth