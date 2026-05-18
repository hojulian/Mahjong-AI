from kfp import dsl, compiler
from kfp import kubernetes

from pipeline.components import train_furo

DATA_PVC = 'data-pvc'
OUTPUT_PVC = 'output-pvc'
IMAGE_PULL_SECRET = 'harbor-creds'
@dsl.pipeline(name='mahjong-furo-training')
def furo_pipeline(
    mode: str = 'chi',
    num_layers: int = 20,
    epochs: int = 10,
    pos_weight: int = 0,
):
    task = train_furo(
        data_dir='/data',
        output_dir=f'/output/checkpoints/{mode}',
        log_dir=f'/output/logs/{mode}',
        mode=mode,
        num_layers=num_layers,
        epochs=epochs,
        pos_weight=pos_weight,
    )
    task.set_cpu_request('4')
    task.set_memory_request('16Gi')
    task.set_gpu_limit('1')
    kubernetes.mount_pvc(task, pvc_name=DATA_PVC, mount_path='/data')
    kubernetes.mount_pvc(task, pvc_name=OUTPUT_PVC, mount_path='/output')
    kubernetes.set_image_pull_secrets(task, [IMAGE_PULL_SECRET])
    kubernetes.empty_dir_mount(task, volume_name='dshm', mount_path='/dev/shm', medium='Memory', size_limit='8Gi')


if __name__ == '__main__':
    compiler.Compiler().compile(furo_pipeline, 'furo_pipeline.yaml')
