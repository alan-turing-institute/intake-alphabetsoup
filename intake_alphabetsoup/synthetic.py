import intake


class SyntheticAlphabetSoupSource(intake.source.base.DataSource):
    container = "ndarray"
    name = "alphabetsoup"
    version = "0.0.1"
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

    def _get_schema(self):
        return intake.source.base.Schema(
            datashape=self.datashape,
            dtype="float32",
            shape=(*self.datashape, 2),
            npartitions=self.image_count,
            extra_metadata=self.metadata,
        )

    def _get_partition(self, i):
        import numpy as np
        import vne.simulate as simulate
        from vne.dataset import SimulatedDataset
        from vne.special.ctf import contrast_transfer_function, convolve_with_ctf

        simulate.set_default_font("HelveticaNeue", 20)

        ctf = contrast_transfer_function(defocus=5e3, box_size=512)

        def preprocessor(x):
            x_ctf = convolve_with_ctf(1 + x, ctf, add_poisson_noise=False,)
            x_noise = x_ctf + np.random.randn(*x.shape) * 40
            return x_noise

        ds = SimulatedDataset(
            preprocessor=preprocessor, simulator=simulate.create_heterogeneous_image
        )

        return ds[0]

    def read(self):
        import numpy as np

        self._load_metadata()

        return np.stack(
            [
                self.read_partition(i)[0].permute(1, 2, 0)
                for i in range(self.image_count)
            ]
        )

    def _close(self):
        # nothing to do
        pass
