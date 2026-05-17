from kfp import dsl, compiler
from kfp import kubernetes

from pipeline.components import train_reward

DATA_PVC = 'data-pvc'
OUTPUT_PVC = 'output-pvc'
IMAGE_PULL_SECRET = 'harbor-creds'


@dsl.pipeline(name='mahjong-reward-training')
def reward_pipeline(
    hidden_dims: int = 50,
    num_layers: int = 2,
    epochs: int = 10,
):
    task = train_reward(
        data_dir='/data',
        output_dir='/output/checkpoints/reward',
        log_dir='/output/logs/reward',
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        epochs=epochs,
    )
    task.set_cpu_request('4')
    task.set_memory_request('16Gi')
    kubernetes.mount_pvc(task, pvc_name=DATA_PVC, mount_path='/data')
    kubernetes.mount_pvc(task, pvc_name=OUTPUT_PVC, mount_path='/output')
    kubernetes.use_image_pull_secret(task, IMAGE_PULL_SECRET)


if __name__ == '__main__':
    compiler.Compiler().compile(reward_pipeline, 'reward_pipeline.yaml')
