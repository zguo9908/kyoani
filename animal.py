# class Animal holds relevant information about a mouse

class Animal:
    def __init__(self, name):
        self.name = name

        self.holding_s_blk = []
        self.holding_l_blk = []
        self.opt_diff_s = []
        self.opt_diff_l = []

        self.perc_rewarded = []
        self.perc_rewarded_s = []  # across days
        self.perc_rewarded_l = []

        self.blk_holding_s = []
        self.blk_holding_l = []

        self.all_holding_s = []
        self.all_holding_l = []

        self.lick_prob_s = []
        self.lick_prob_l = []

        self.holding_perf_s = []
        self.holding_perf_l = []

        self.blk_miss_perc_s = []
        self.blk_miss_perc_l = []
