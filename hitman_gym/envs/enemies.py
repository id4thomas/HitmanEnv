# enemy agents

#Fixed Enemy
class BlueEnemy:
    def __init__(self, r, c, dir, conn):
        # direction
        # location
        self.pos = [r, c]
        self.dir = dir
        self.conn = "{0:4b}".format(int(conn))[-4:]
        #print("conn str: {} conn {}".format(conn,self.conn))
        # Up,down,left,right
        self.check_r = [-1, 1, 0, 0]
        self.check_c = [0, 0, -1, 1]

    def check_range(self, r, c):
        r_chk = (r - self.pos[0] == self.check_r[self.dir])
        c_chk = (c - self.pos[1] == self.check_c[self.dir])

        is_conn=self.conn[self.dir]=='1'
        return r_chk and c_chk and is_conn

    def check_caught(self, r, c):
        return self.pos[0] == r and self.pos[1] == c

#Moving Enemy
class YellowEnemy:
    def __init__(self, r, c, dir):
        # direction
        # location
        self.pos = [r, c]
        self.dir = dir
        #self.conn = "{0:4b}".format(int(conn))[-4:]
        #print("conn str: {} conn {}".format(conn,self.conn))
        # Up,down,left,right
        self.check_r = [-1, 1, 0, 0]
        self.check_c = [0, 0, -1, 1]

    def check_range(self, r, c, conn):
        r_chk = (r - self.pos[0] == self.check_r[self.dir])
        c_chk = (c - self.pos[1] == self.check_c[self.dir])
        conn="{0:4b}".format(int(conn))[-4:]
        is_conn=conn[self.dir]=='1'
        return r_chk and c_chk and is_conn

    def check_caught(self, r, c):
        return self.pos[0] == r and self.pos[1] == c

    def moved_pos(self,pos):
        return [pos[0]+self.check_r[self.dir],pos[1]+self.check_c[self.dir]]

    #def update_dir(self,dir):
        