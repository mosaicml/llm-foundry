class ScalingConfig:
    def __init__(self, gpusNum: int, gpuType: str, poolName: str):
        # TODO: validate the inputs
        self.gpusNum = gpusNum
        self.gpuType = gpuType
        self.poolName = poolName

    @property
    def toCompute(self):
        return {
            'gpus': self.gpusNum,
            'gpu_type': self.gpuType,
            'cluster': self.poolName
        }
