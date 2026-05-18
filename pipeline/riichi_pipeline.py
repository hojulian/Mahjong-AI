from kfp import dsl, compiler
from kfp import kubernetes

from pipeline.components import train_riichi

DATA_PVC = 'data-pvc'
OUTPUT_PVC = 'output-pvc'
IMAGE_PULL_SECRET = 'harbor-creds'


@dsl.pipeline(name='mahjong-riichi-training')
def riichi_pipeline(
    num_layers: int = 20,
    epochs: int = 10,
    pos_weight: int = 0,
):
    task = train_riichi(
        data_dir='/data',
        output_dir='/output/checkpoints/riichi',
        log_dir='/output/logs/riichi',
        num_layers=num_layers,
        epochs=epochs,
        pos_weight=pos_weight,
    )
    task.set_cpu_request('4')
    task.set_memory_request('16Gi')
    kubernetes.mount_pvc(task, pvc_name=DATA_PVC, mount_path='/data')
    kubernetes.mount_pvc(task, pvc_name=OUTPUT_PVC, mount_path='/output')
    kubernetes.set_image_pull_secrets(task, [IMAGE_PULL_SECRET])


if __name__ == '__main__':
    compiler.Compiler().compile(riichi_pipeline, 'riichi_pipeline.yaml')
