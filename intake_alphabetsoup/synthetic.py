class SyntheticAlphabetSoupSource(intake.source.base.DataSource):
    container = "ndarray"
    name = "alphabetsoup"
    version = "0.0.1"
    partition_access = True

    def __init__(
        self, datashape=(512, 512), ctf_defocus=5e3, ctf_box_size=512, metadata=None
    ):
        super().__init__(metadata=metadata)

    def _get_schema(self):
        import vne

        return intake.source.base.Schema(
            datashape=datashape,
            dtype="int64",
            shape=datashape + (2,),
            npartitions=2,
            extra_metadata=metadata,
        )

    def _get_partition(self, i):
        return

    def read(self):
        self._load_metadata()
        return pd.concat([self.read_partition(i) for i in range(self.npartitions)])

    def _close(self):
        # nothing to do
        pass
