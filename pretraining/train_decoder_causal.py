import importlib.util
import os


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "pretraining", "decoder", "train.py")


def load_training_module():
    spec = importlib.util.spec_from_file_location("decoder_causal_train", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    train = load_training_module()
    train.OUTPUT_DIR = os.environ.get(
        "BABYLM_MODEL_OUTPUT_DIR",
        os.path.join(WORKSPACE_ROOT, "models", "decoder-causal"),
    )
    train.LOGS_DIR = os.environ.get(
        "BABYLM_LOGS_DIR",
        os.path.join(WORKSPACE_ROOT, "logs", "decoder_causal_pretrain"),
    )
    train.main()
