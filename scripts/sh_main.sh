CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--config ddim_cifar10.yml \
--model DDIM \
--model_path /cw/working-eowyn/mingxiao/TS-DPM/models/ddim_cifar10.ckpt \
--method DDIM \
--sample_speed 50 \

