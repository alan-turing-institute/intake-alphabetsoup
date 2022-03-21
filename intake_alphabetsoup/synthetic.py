import intake
from functools import lru_cache
from . import __version__


class SyntheticAlphabetSoupSource(intake.source.base.DataSource):
    container = "python"
    name = "alphabetsoup"
    version = __version__
    partition_access = True

    def __init__(
        self,
        datashape=(512, 512),
        image_count=1,
        ctf_defocus=5e3,
        ctf_box_size=512,
        metadata=None,
    ):
        super().__init__(metadata=metadata)

        self.datashape = datashape
        self.image_count = image_count
        self.ctf_defocus = ctf_defocus
        self.ctf_box_size = ctf_box_size

        self._ds = None

    def _get_schema(self):
        import numpy as np
        import vne.simulate as simulate
        from vne.dataset import SimulatedDataset
        from vne.special.ctf import contrast_transfer_function, convolve_with_ctf

        if self._ds is None:
            simulate.set_default_font("HelveticaNeue", 20)

            ctf = contrast_transfer_function(
                defocus=self.ctf_defocus, box_size=self.ctf_box_size
            )

            def preprocessor(x):
                x_ctf = convolve_with_ctf(1 + x, ctf, add_poisson_noise=False,)
                x_noise = x_ctf + np.random.randn(*x.shape) * 40
                return x_noise

            self._ds = SimulatedDataset(
                preprocessor=preprocessor,
                simulator=simulate.create_heterogeneous_image,
                size=self.datashape,
            )

        return intake.source.base.Schema(
            dtype="float32",
            shape=None,
            datashape=None,
            npartitions=self.image_count,
            extra_metadata=self.metadata,
        )

    @lru_cache(maxsize=None)
    def _get_partition(self, i):
        self._load_metadata()

        data = self._ds[i]
        images = data[0].permute(1, 2, 0)
        boxes = data[1]["boxes"]
        labels = data[1]["labels"]

        return (images, boxes, labels)

    def read(self):
        self._load_metadata()

        # return tuple of lists, rather than list of tuples
        gen_data = (self.read_partition(i) for i in range(self.image_count))

        return tuple(map(list, zip(*gen_data)))

    def _close(self):
        self._ds = None
