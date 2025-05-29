unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
# 设置一个基准端口变量
export BASE_PORT=${BASE_PORT:-14181}

# 使用基准端口变量进行偏移
export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}

# 基于 BASE_PORT 变量依次递增端口号
# export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-$((BASE_PORT))}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-$((BASE_PORT + 1))}
export SAM_PORT=${SAM_PORT:-$((BASE_PORT + 2))}
# export YOLOV7_PORT=${YOLOV7_PORT:-$((BASE_PORT + 3))}
export OwlV2_PORT=${OwlV2_PORT:-$((BASE_PORT + 4))}

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Run commands in each pane
tmux send-keys -t ${session_name}:0.1 "${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "${VLFM_PYTHON} -m vlfm.vlm.owl_vit_test --port ${OwlV2_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
