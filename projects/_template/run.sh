accelerate launch \
    --num_machines $WORLD_SIZE \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $((8 * WORLD_SIZE)) \
    "$(dirname "$(realpath "$0")")/run.py" \
    -c configs.debug
