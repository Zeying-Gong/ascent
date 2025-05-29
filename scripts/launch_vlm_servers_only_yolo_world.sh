export CUDA_VISIBLE_DEVICES=1
# 设置一个基准端口变量
export BASE_PORT=${BASE_PORT:-13181}

# 使用基准端口变量进行偏移
export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
# export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
# export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
# 从环境变量读取 YOLO_WORLD_DIR，若未设置则使用默认路径
export YOLO_WORLD_DIR=${YOLO_WORLD_DIR:-"/home/zeyingg/github/habitat-lab-vlfm/YOLO-World"}

# 使用 YOLO_WORLD_DIR 构建 YOLO_WORLD_CONFIG 和 YOLO_WORLD_WEIGHTS 的绝对路径
export YOLO_WORLD_CONFIG="${YOLO_WORLD_DIR}/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minval.py"
export YOLO_WORLD_WEIGHTS="${YOLO_WORLD_DIR}/weights/yolo_world_v2_l_obj365v1_goldg_cc3mv2_pretrain-2f3a4a22.pth"

export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}

# 基于 BASE_PORT 变量依次递增端口号
# export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-$((BASE_PORT))}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-$((BASE_PORT + 1))}
export SAM_PORT=${SAM_PORT:-$((BASE_PORT + 2))}
# export YOLOV7_PORT=${YOLOV7_PORT:-$((BASE_PORT + 3))}
export YOLO_WORLD_PORT=${YOLO_WORLD_PORT:-$((BASE_PORT + 4))}

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Run commands in each pane
# tmux send-keys -t ${session_name}:0.0 "${VLFM_PYTHON} -m vlfm.vlm.grounding_dino_test --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m
# tmux send-keys -t ${session_name}:0.3 "${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "${VLFM_PYTHON} -m vlfm.vlm.yolo_world_test --port ${YOLO_WORLD_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
