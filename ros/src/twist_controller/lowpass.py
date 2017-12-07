
class LowPassFilter(object):
    def __init__(self, num_val):
	self.num_val = num_val
	self.list_val = []

    def filt(self, val):
	self.list_val.insert(0, val)
	if (len(self.list_val) == self.num_val+1):
	    self.list_val.pop()
	return sum(self.list_val) / self.num_val

	
