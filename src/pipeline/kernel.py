from enum import Enum

from vtkmodules.all import vtkInterpolationKernel, vtkGaussianKernel

class KernelType(Enum):
    GAUSSIAN = 'GAUSSIAN'

def get_kernel(kernel_type: KernelType, **kwargs) -> vtkInterpolationKernel:
    if kernel_type == KernelType.GAUSSIAN:
        kernel = vtkGaussianKernel()
        kernel


