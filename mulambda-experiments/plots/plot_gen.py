import matplotlib.pyplot as plt


class ExperimentPlot:
    @property
    def title(self):
        return "Experiment Plot"

    @property
    def xlabel(self):
        return "X Label"

    @property
    def ylabel(self):
        return "Y Label"

    def _do_plot(self):
        raise NotImplementedError

    def plot(self, output_dir: str, format: str = "pdf"):
        self._do_plot()
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend()
        plt.savefig(f"{output_dir}/{self.title}.{format}")
        plt.clf()


class ECDFPlot(ExperimentPlot):

    def title(self):
        return "ECDF Plot"
