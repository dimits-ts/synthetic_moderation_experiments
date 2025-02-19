# Function to check if a file exists and download if it doesn't
function download_if_not_exists() {
    local file_path=$1
    local download_url=$2

    # Check if the file exists
    if [ ! -f "$file_path" ]; then
        echo "File $file_path does not exist. Downloading..."
        download_from_huggingface "$file_path" "$download_url"
    else
        echo "File $file_path already exists. Skipping download."
    fi
}

# Function to handle downloading from Huggingface
function download_from_huggingface() {
    local file_path=$1
    local download_url=$2

    wget "$download_url" -O "$file_path"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR="$(dirname "$SCRIPT_DIR")"

# Create the "models" subdirectory, if it does not exist
MODELS_DIR="$SRC_DIR/models"
if [ ! -d "$MODELS_DIR" ]; then
    mkdir -p "$MODELS_DIR"
fi

#download_if_not_exists "$SRC_DIR/models/llama-3-8B-instruct.gguf" "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true"
download_if_not_exists "$SRC_DIR/models/llama-3.2-8B-instruct.gguf" "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_V2_F16.gguf?download=true"