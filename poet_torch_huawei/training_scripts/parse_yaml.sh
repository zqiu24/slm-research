#!/bin/bash

# ==============================================================================
# YAML配置解析脚本 (YAML Configuration Parser)
# ==============================================================================
# 用法 (Usage):
#   source parse_yaml.sh <yaml_config_path>
#
# 功能 (Function):
#   解析YAML配置文件中的MODEL_ARGS部分，并将其转换为命令行参数
#   Parse MODEL_ARGS from YAML config and convert to command-line arguments
#
# 输出 (Output):
#   设置全局变量 MODEL_ARGS_FROM_CONFIG，包含解析后的参数字符串
#   Sets global variable MODEL_ARGS_FROM_CONFIG with parsed argument string
# ==============================================================================

# 获取YAML配置文件路径 (第一个参数)
# Get YAML config file path (first parameter)
YAML_CONFIG_PATH="${1:-}"

if [ -z "$YAML_CONFIG_PATH" ]; then
    echo "Error: YAML config path not provided" >&2
    echo "Usage: source parse_yaml.sh <yaml_config_path>" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$YAML_CONFIG_PATH" ]; then
    echo "Error: YAML config file not found at: $YAML_CONFIG_PATH" >&2
    return 1 2>/dev/null || exit 1
fi

# 检查Python是否可用
# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH" >&2
    return 1 2>/dev/null || exit 1
fi

# 检查PyYAML是否安装
# Check if PyYAML is installed
if ! python3 -c "import yaml" &> /dev/null; then
    echo "Error: PyYAML is not installed. Please install it:" >&2
    echo "  pip install pyyaml" >&2
    return 1 2>/dev/null || exit 1
fi

# 使用Python解析YAML配置
# Parse YAML config using Python
MODEL_ARGS_FROM_CONFIG=$(python3 -c '
import sys
import yaml

# 获取YAML配置文件路径
config_path = sys.argv[1]

try:
    with open(config_path, "r") as f:
        # 安全加载YAML
        data = yaml.safe_load(f)
        # 获取MODEL_ARGS部分
        args = data.get("MODEL_ARGS", {})

    cmd_list = []

    # 遍历并处理每个参数
    for k, v in args.items():
        # 统一转为字符串并小写，以便同时处理 True/False 和 "true"/"false"
        str_v = str(v).lower()

        if v is False or str_v == "false":
            # 忽略False值的参数
            continue
        elif v is True or str_v == "true":
            # True值的参数只保留Key (作为开关)
            cmd_list.append(k)
        else:
            # 普通键值对
            cmd_list.append(f"{k} {v}")

    # 输出结果
    print(" ".join(cmd_list))

except FileNotFoundError:
    print(f"Error: Config file {config_path} not found.", file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

' "$YAML_CONFIG_PATH")

# 检查解析是否成功
# Check if parsing was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse YAML config" >&2
    return 1 2>/dev/null || exit 1
fi

# 导出变量供调用脚本使用
# Export variable for calling script
export MODEL_ARGS_FROM_CONFIG
