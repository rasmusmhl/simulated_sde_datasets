from .dataset import Dataset
from .utils import dataloader

from .diffusion_data import (
    OrnsteinUhlenbeckDiffusionDataset,
    StudentDiffusionDataset,
)
from .lotka_volterra_data import (
    StochasticLotkaVolterraDataset,
    StudentStochasticLotkaVolterraDataset,
)
from .stochastic_lorenz_data import (
    StochasticLorenzAttractorDataset,
    StudentStochasticLorenzAttractorDataset,
)

from .graph_diffusion_data import GraphStudentDiffusionDataset

from .forcing_functions import (
    SineForcingMixin,
    SquarePulseForcingMixin,
    TriangleDropForcingMixin,
)


class SineForcingStudentDiffusionDataset(SineForcingMixin, StudentDiffusionDataset): ...


class SineForcingOUDiffusionDataset(
    SineForcingMixin, OrnsteinUhlenbeckDiffusionDataset
): ...


class SquareForcingStudentDiffusionDataset(
    SquarePulseForcingMixin, StudentDiffusionDataset
): ...


class SquareForcingOUDiffusionDataset(
    SquarePulseForcingMixin, OrnsteinUhlenbeckDiffusionDataset
): ...


class TriangleDropForcingOUDiffusionDataset(
    TriangleDropForcingMixin, OrnsteinUhlenbeckDiffusionDataset
): ...


class TriangleDropForcingStudentDiffusionDataset(
    TriangleDropForcingMixin, StudentDiffusionDataset
): ...
