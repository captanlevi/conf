class Logger:
    def __init__(self,name = "",verbose = True,**kwargs):
        """
        kwargs is a a dict mapping name of metric with an empty list or some values in case or resuming

        Logger populates each with [value1,value2, ....]
        """
        self.name = name
        self.verbose = verbose
        self.metrices_dict = kwargs
        self.metric_report_step_size = dict()
        self.default_step_size = 100

    def addMetric(self,metric_name,value):
        if metric_name not in self.metrices_dict:
            self.metrices_dict[metric_name] = []
        self.metrices_dict[metric_name].append(value)
        if self.verbose:
            self.reporting(metric_name= metric_name)
       
    def setMetricReportSteps(self,metric_name,step_size):
        self.metric_report_step_size[metric_name] = step_size

    
    def getMetric(self,metric_name):
        return self.metrices_dict[metric_name]
    
    def getAllMetricNames(self):
        return list(self.metrices_dict.keys())
    



    def reporting(self,metric_name):
        """
        reports the mean of the last step_size of the metric
        if the the length%step_size is 0
        """
        step_size = self.metric_report_step_size[metric_name] if metric_name in self.metric_report_step_size else self.default_step_size

        if len(self.metrices_dict[metric_name]) % step_size == 0:
            values = self.metrices_dict[metric_name][-step_size:]
            print("{} ---- {} metric {} = {}".format(self.name,len(self.metrices_dict[metric_name]),metric_name,sum(values)/len(values)))

        