from kfp import dsl

BASE_IMAGE = 'harbor.macaroni-tet.ts.net/mahjong-ai/training:latest'


@dsl.component(base_image=BASE_IMAGE)
def train_discard(
    data_dir: str,
    output_dir: str,
    log_dir: str,
    num_layers: int = 50,
    epochs: int = 10,
    batch_size: int = 512,
):
    import subprocess, sys
    subprocess.run([
        sys.executable, '/app/sl_train/train_discard_model.py',
        '--num_layers', str(num_layers),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--data-dir', data_dir,
        '--output-dir', output_dir,
        '--log-dir', log_dir,
    ], check=True)


@dsl.component(base_image=BASE_IMAGE)
def train_riichi(
    data_dir: str,
    output_dir: str,
    log_dir: str,
    num_layers: int = 20,
    epochs: int = 10,
    pos_weight: int = 0,
):
    import subprocess, sys
    cmd = [
        sys.executable, '/app/sl_train/train_riichi_model.py',
        '--num_layers', str(num_layers),
        '--epochs', str(epochs),
        '--data-dir', data_dir,
        '--output-dir', output_dir,
        '--log-dir', log_dir,
    ]
    if pos_weight > 0:
        cmd += ['--pos_weight', str(pos_weight)]
    subprocess.run(cmd, check=True)


@dsl.component(base_image=BASE_IMAGE)
def train_furo(
    data_dir: str,
    output_dir: str,
    log_dir: str,
    mode: str = 'chi',
    num_layers: int = 20,
    epochs: int = 10,
    pos_weight: int = 0,
):
    import subprocess, sys
    cmd = [
        sys.executable, '/app/sl_train/train_furo_model.py',
        '--mode', mode,
        '--num_layers', str(num_layers),
        '--epochs', str(epochs),
        '--data-dir', data_dir,
        '--output-dir', output_dir,
        '--log-dir', log_dir,
    ]
    if pos_weight > 0:
        cmd += ['--pos_weight', str(pos_weight)]
    subprocess.run(cmd, check=True)


@dsl.component(base_image=BASE_IMAGE)
def train_reward(
    data_dir: str,
    output_dir: str,
    log_dir: str,
    hidden_dims: int = 50,
    num_layers: int = 2,
    epochs: int = 10,
):
    import subprocess, sys
    subprocess.run([
        sys.executable, '/app/sl_train/train_reward.py',
        '--hidden_dims', str(hidden_dims),
        '--num_layers', str(num_layers),
        '--epochs', str(epochs),
        '--data-dir', data_dir,
        '--output-dir', output_dir,
        '--log-dir', log_dir,
    ], check=True)
