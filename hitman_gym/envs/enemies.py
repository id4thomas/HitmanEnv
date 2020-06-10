# enemy agents


class BlueEnemy:
    def __init__(self, r, c, dir, conn):
        # direction
        # location
        self.pos = [r, c]
        self.dir = dir - 3
        self.conn = "{0:4b}".format(int(conn))[-4:]
        #print("conn str: {} conn {}".format(conn,self.conn))
        # Up,down,left,right
        self.check_r = [-1, 1, 0, 0]
        self.check_c = [0, 0, -1, 1]

    def check_range(self, r, c):
        r_chk = (r - self.pos[0] == self.check_r[self.dir])
        c_chk = (c - self.pos[1] == self.check_c[self.dir])
        # Need to implement if connected check
        is_conn=self.conn[self.dir]=='1'
        return r_chk and c_chk and is_conn

    def check_caught(self, r, c):
        return self.pos[0] == r and self.pos[1] == c
