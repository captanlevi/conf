from typing import List

class FlowConfig:
    def __init__(self,grain : float, band_thresholds : List[float]):
        """
        Configuration of a flowdata, all unit of time measurement is second
        grain - data interval in seconds
        band_thresholds - thresholds for splitting flowprint array ex [0.1250] or [1250]
        """
        assert grain > 0, "Nonsense"
        self.grain = grain
        self.band_thresholds = band_thresholds

        

    def __eq__(self, __value: object) -> bool:
        if self.grain == __value.grain and self.band_thresholds == __value.band_thresholds:
            return True
        return False
