export CUDA_VISIBLE_DEVICES=2

python ../tools/run_net.py --config-file ../projects/neus/configs/dtu_trainable/neus_dtu24_womask_two_optim_0.py --type self_calibrated --task train