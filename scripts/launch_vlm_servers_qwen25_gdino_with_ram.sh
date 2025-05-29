

# 设置一个基准端口变量
export BASE_PORT=${BASE_PORT:-12181}

# 使用基准端口变量进行偏移
export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}

export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}

export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export RAM_CHECKPOINT=${RAM_CHECKPOINT:-data/ram_plus_swin_large_14m.pth}
# 基于 BASE_PORT 变量依次递增端口号
export QWEN2_5ITM_PORT=${QWEN2_5ITM_PORT:-$((BASE_PORT))}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-$((BASE_PORT + 1))}
export SAM_PORT=${SAM_PORT:-$((BASE_PORT + 2))}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-$((BASE_PORT + 3))}
export RAM_PORT=${RAM_PORT:-$((BASE_PORT + 4))}
export YOLOV7_PORT=$((BASE_PORT + 5))

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# 将每一行水平分割为 2 列
tmux split-window -v -t ${session_name}:0
tmux split-window -v -t ${session_name}:0

# 将窗口垂直分割为 3 行
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2
tmux split-window -h -t ${session_name}:0.4

# # 将窗口水平分割为 3 列
# tmux split-window -h -t ${session_name}:0
# tmux split-window -h -t ${session_name}:0

# # 将每一列垂直分割为 2 行
# tmux split-window -v -t ${session_name}:0.0
# tmux split-window -v -t ${session_name}:0.1
# tmux split-window -v -t ${session_name}:0.2

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "export CUDA_VISIBLE_DEVICES=0; ${VLFM_PYTHON} -m vlfm.vlm.qwen25itm --port ${QWEN2_5ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "export CUDA_VISIBLE_DEVICES=1; ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "export CUDA_VISIBLE_DEVICES=1; ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "export CUDA_VISIBLE_DEVICES=1; ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino_test --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.4 "export CUDA_VISIBLE_DEVICES=1; ${VLFM_PYTHON} -m vlfm.vlm.ram_test --port ${RAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.5 "export CUDA_VISIBLE_DEVICES=1; ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
