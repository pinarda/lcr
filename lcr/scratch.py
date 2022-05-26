class Test:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        self.d = 4
        self.e = 5
        self.f = 6
        self.g = 7
        self.h = 8
        self.i = 9
        self.j = 10
        self.k = 11
        self.l = 12
        self.m = 13
        self.n = 14
        self.o = 15
        self.p = 16
        self.q = 17
        self.r = 18
        self.s = 19
        self.t = 20
        self.u = 21
        self.v = 22
        self.w = 23
        self.x = 24
        self.y = 25
        self.z = 26
        self.aa = 27
        self.bb = 28
        self.cc = 29
        self.dd = 30
        self.ee = 31
        self.ff = 32
        self.gg = 33
        self.hh = 34
        self.ii = 35
        self.jj = 36
        self.kk = 37
        self.ll = 38
        self.mm = 39
        self.nn = 40
        self.oo = 41
        self.pp = 42
        self.qq = 43
        self.rr = 44
        self.ss = 45
        self.tt = 46
        self.uu = 47
        self.vv = 48
        self.ww = 49
        self.xx = 50
        self.yy = 51
        self.zz = 52
        self.aaa = 53
        self.bbb = 54
        self.ccc = 55
        self.ddd = 56
        self.eee = 57
        self.fff = 58
        self.ggg = 59
        self.hhh = 60
        self.iii = 61
        self.jjj = 62

    # find all images without alternate text
    # and give them a red border
    def find_no_alt(self):
        for img in self.soup.find_all('img'):
            if img.get('alt') == '':
                img['style'] = 'border: 2px solid red'
