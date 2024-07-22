export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

script=$1
shift
bash install.sh

echo "running"
bash $script $@
echo "finish"
