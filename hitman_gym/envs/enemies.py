#enemy agents

class BlueEnemy:
    def __init__(self,r,c,dir):
        #direction
        #location
        self.pos=[r,c]
        self.dir=dir-3

        #Up,down,left,right
        self.check_r=[-1,1,0,0]
        self.check_c=[0,0,-1,1]

    def check_range(self,r,c):
        r_chk=(r-self.pos[0]==self.check_r[self.dir])
        c_chk=(c-self.pos[1]==self.check_c[self.dir])
        return r_chk and c_chk

    def check_caught(self,r,c):
        return self.pos[0]==r and self.pos[1]==c
# pycharm github 연동 test :^) == HI