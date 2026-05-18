from kfp import dsl, compiler
from kfp import kubernetes

from pipeline.components import train_discard

DATA_PVC = 'data-pvc'
OUTPUT_PVC = 'output-pvc'
IMAGE_PULL_SECRET = 'harbor-creds'
@dsl.pipeline(name='mahjong-discard-training')
def discard_pipeline(
    num_layers: int = 50,
    epochs: int = 10,
    batch_size: int = 512,
):
    task = train_discard(
        data_dir='/data',
        output_dir='/output/checkpoints/discard',
        log_dir='/output/logs/discard',
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
    )
    task.set_cpu_request('4')
    task.set_memory_request('16Gi')
    task.set_gpu_limit('1')
    kubernetes.mount_pvc(task, pvc_name=DATA_PVC, mount_path='/data')
    kubernetes.mount_pvc(task, pvc_name=OUTPUT_PVC, mount_path='/output')
    kubernetes.set_image_pull_secrets(task, [IMAGE_PULL_SECRET])
    kubernetes.empty_dir_mount(task, volume_name='dshm', mount_path='/dev/shm', medium='Memory', size_limit='8Gi')


if __name__ == '__main__':
    compiler.Compiler().compile(discard_pipeline, 'discard_pipeline.yaml')
