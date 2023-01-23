from diive.pkgs.outlierdetection.seasonaltrend import OutlierSTLIQR


class FluxOutlierRemoval:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # todo use metadata_df?

    def run(self):
        stl = OutlierSTLIQR(**self.kwargs)
        stl.run()


def example():
    pass


if __name__ == '__main__':
    example()
