class BatchReplacer:
    def __init__(self,mx_ratio,lim_steps):
        self.mx_ratio = mx_ratio
        self.lim_ratio = lim_steps
    

    def getFraction(self,step):
        return min(self.mx_ratio,step/self.lim_ratio)
    