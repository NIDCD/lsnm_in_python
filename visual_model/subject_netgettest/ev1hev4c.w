% Tue May 24 15:52:21 2016

% Input Layer: (9, 9)
% Output Layer: (9, 9)
% Fanout Size: (3, 3)

Connect(ev1h, ev4c)  {
  From:  (1, 1)  {
    |              |     |              |     |              | 
    ([ 1, 9]  0.024676)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (1, 2)  {
    |              |     |              |     |              | 
    ([ 1, 1]  0.021285)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (1, 3)  {
    |              |     ([ 9, 3]  0.019624)     |              | 
    |              |     |              |     ([ 1, 4]  0.029879) 
    |              |     ([ 2, 3]  0.018375)     |              | 
  }
  From:  (1, 4)  {
    |              |     |              |     |              | 
    ([ 1, 3]  0.028616)     ([ 1, 4]  0.026505)     |              | 
    |              |     |              |     |              | 
  }
  From:  (1, 5)  {
    |              |     |              |     |              | 
    |              |     ([ 1, 5]  0.027739)     ([ 1, 6]  0.021164) 
    |              |     ([ 2, 5]  0.029904)     |              | 
  }
  From:  (1, 6)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     ([ 2, 6]  0.016822)     |              | 
  }
  From:  (1, 7)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    ([ 1, 1]  0.009220)   }
  From:  (1, 8)  {
    |              |     ([ 9, 8]  0.022732)     |              | 
    ([ 1, 7]  0.013328)     ([ 1, 8]  0.024728)     |              | 
    |              |     ([ 2, 8]  0.019114)     |              | 
  }
  From:  (1, 9)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     ([ 2, 9]  0.024059)     |              | 
  }
  From:  (2, 1)  {
    |              |     |              |     |              | 
    ([ 2, 9]  0.011297)     ([ 2, 1]  0.015623)     ([ 2, 2]  0.015035) 
    |              |     ([ 3, 1]  0.021033)     |              | 
  }
  From:  (2, 2)  {
    |              |     ([ 1, 2]  0.025141)     |              | 
    |              |     ([ 2, 2]  0.016978)     ([ 2, 3]  0.014565) 
    |              |     ([ 3, 2]  0.029675)     |              | 
  }
  From:  (2, 3)  {
    |              |     ([ 1, 3]  0.024311)     |              | 
    |              |     ([ 2, 3]  0.027655)     ([ 2, 4]  0.019598) 
    |              |     ([ 3, 3]  0.021146)     |              | 
  }
  From:  (2, 4)  {
    |              |     ([ 1, 4]  0.025621)     |              | 
    |              |     |              |     |              | 
    |              |     ([ 3, 4]  0.010243)     |              | 
  }
  From:  (2, 5)  {
    |              |     ([ 1, 5]  0.018954)     |              | 
    |              |     ([ 2, 5]  0.018627)     |              | 
    |              |     |              |     |              | 
  }
  From:  (2, 6)  {
    |              |     ([ 1, 6]  0.014038)     |              | 
    ([ 2, 5]  0.011807)     ([ 2, 6]  0.023124)     |              | 
    |              |     |              |     |              | 
  }
  From:  (2, 7)  {
    |              |     ([ 1, 7]  0.016289)     |              | 
    ([ 2, 6]  0.020358)     ([ 2, 7]  0.017498)     |              | 
    |              |     ([ 3, 7]  0.028747)     |              | 
  }
  From:  (2, 8)  {
    |              |     |              |     |              | 
    ([ 2, 7]  0.026436)     ([ 2, 8]  0.029784)     |              | 
    |              |     ([ 3, 8]  0.024622)     |              | 
  }
  From:  (2, 9)  {
    |              |     |              |     |              | 
    ([ 2, 8]  0.015660)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (3, 1)  {
    |              |     ([ 2, 1]  0.013602)     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (3, 2)  {
    |              |     |              |     |              | 
    ([ 3, 1]  0.013974)     |              |     ([ 3, 3]  0.010668) 
    |              |     ([ 4, 2]  0.010762)     |              | 
  }
  From:  (3, 3)  {
    |              |     |              |     |              | 
    ([ 3, 2]  0.011033)     |              |     |              | 
    |              |     ([ 4, 3]  0.021227)     |              | 
  }
  From:  (3, 4)  {
    |              |     ([ 2, 4]  0.013028)     |              | 
    ([ 3, 3]  0.014262)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (3, 5)  {
    |              |     |              |     |              | 
    ([ 3, 4]  0.021901)     |              |     ([ 3, 6]  0.013299) 
    |              |     ([ 4, 5]  0.025311)     |              | 
  }
  From:  (3, 6)  {
    |              |     ([ 2, 6]  0.019100)     |              | 
    ([ 3, 5]  0.012608)     |              |     |              | 
    |              |     ([ 4, 6]  0.022512)     |              | 
  }
  From:  (3, 7)  {
    |              |     ([ 2, 7]  0.028533)     |              | 
    ([ 3, 6]  0.023335)     ([ 3, 7]  0.021697)     ([ 3, 8]  0.020941) 
    |              |     ([ 4, 7]  0.026713)     |              | 
  }
  From:  (3, 8)  {
    |              |     ([ 2, 8]  0.025401)     |              | 
    ([ 3, 7]  0.023916)     ([ 3, 8]  0.023687)     ([ 3, 9]  0.026616) 
    |              |     |              |     |              | 
  }
  From:  (3, 9)  {
    |              |     ([ 2, 9]  0.012355)     |              | 
    |              |     |              |     |              | 
    |              |     ([ 4, 9]  0.019690)     |              | 
  }
  From:  (4, 1)  {
    |              |     ([ 3, 1]  0.027278)     |              | 
    ([ 4, 9]  0.016970)     ([ 4, 1]  0.027913)     ([ 4, 2]  0.023107) 
    |              |     ([ 5, 1]  0.013321)     |              | 
  }
  From:  (4, 2)  {
    |              |     ([ 3, 2]  0.026332)     |              | 
    |              |     ([ 4, 2]  0.026497)     |              | 
    |              |     ([ 5, 2]  0.025928)     |              | 
  }
  From:  (4, 3)  {
    |              |     |              |     |              | 
    ([ 4, 2]  0.021324)     ([ 4, 3]  0.026628)     ([ 4, 4]  0.010465) 
    |              |     ([ 5, 3]  0.019070)     |              | 
  }
  From:  (4, 4)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    ([ 1, 1]  0.008342)   }
  From:  (4, 5)  {
    |              |     ([ 3, 5]  0.014055)     |              | 
    ([ 4, 4]  0.025121)     ([ 4, 5]  0.015937)     |              | 
    |              |     |              |     |              | 
  }
  From:  (4, 6)  {
    |              |     ([ 3, 6]  0.011379)     |              | 
    ([ 4, 5]  0.025332)     ([ 4, 6]  0.018379)     |              | 
    |              |     ([ 5, 6]  0.022460)     |              | 
  }
  From:  (4, 7)  {
    |              |     ([ 3, 7]  0.013722)     |              | 
    ([ 4, 6]  0.027887)     ([ 4, 7]  0.029106)     |              | 
    |              |     ([ 5, 7]  0.016084)     |              | 
  }
  From:  (4, 8)  {
    |              |     ([ 3, 8]  0.014593)     |              | 
    |              |     |              |     ([ 4, 9]  0.024346) 
    |              |     ([ 5, 8]  0.026800)     |              | 
  }
  From:  (4, 9)  {
    |              |     |              |     |              | 
    ([ 4, 8]  0.021503)     ([ 4, 9]  0.021392)     ([ 4, 1]  0.012329) 
    |              |     |              |     |              | 
  }
  From:  (5, 1)  {
    |              |     ([ 4, 1]  0.011456)     |              | 
    |              |     |              |     ([ 5, 2]  0.022501) 
    |              |     |              |     |              | 
  }
  From:  (5, 2)  {
    |              |     ([ 4, 2]  0.017694)     |              | 
    ([ 5, 1]  0.022929)     |              |     |              | 
    |              |     ([ 6, 2]  0.022323)     |              | 
  }
  From:  (5, 3)  {
    |              |     |              |     |              | 
    |              |     ([ 5, 3]  0.011982)     |              | 
    |              |     |              |     |              | 
  }
  From:  (5, 4)  {
    |              |     ([ 4, 4]  0.010929)     |              | 
    ([ 5, 3]  0.022403)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (5, 5)  {
    |              |     |              |     |              | 
    |              |     |              |     ([ 5, 6]  0.025214) 
    |              |     |              |     |              | 
  }
  From:  (5, 6)  {
    |              |     |              |     |              | 
    ([ 5, 5]  0.026619)     |              |     ([ 5, 7]  0.015719) 
    |              |     ([ 6, 6]  0.011186)     |              | 
  }
  From:  (5, 7)  {
    |              |     ([ 4, 7]  0.022218)     |              | 
    |              |     ([ 5, 7]  0.023790)     ([ 5, 8]  0.022676) 
    |              |     |              |     |              | 
  }
  From:  (5, 8)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    ([ 1, 1]  0.001695)   }
  From:  (5, 9)  {
    |              |     ([ 4, 9]  0.022217)     |              | 
    ([ 5, 8]  0.019289)     ([ 5, 9]  0.010311)     ([ 5, 1]  0.019820) 
    |              |     ([ 6, 9]  0.010552)     |              | 
  }
  From:  (6, 1)  {
    |              |     ([ 5, 1]  0.015469)     |              | 
    ([ 6, 9]  0.029419)     |              |     ([ 6, 2]  0.019185) 
    |              |     ([ 7, 1]  0.013765)     |              | 
  }
  From:  (6, 2)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    ([ 1, 1]  0.003686)   }
  From:  (6, 3)  {
    |              |     |              |     |              | 
    ([ 6, 2]  0.014354)     ([ 6, 3]  0.024801)     ([ 6, 4]  0.025120) 
    |              |     |              |     |              | 
  }
  From:  (6, 4)  {
    |              |     ([ 5, 4]  0.019089)     |              | 
    |              |     |              |     ([ 6, 5]  0.029129) 
    |              |     |              |     |              | 
  }
  From:  (6, 5)  {
    |              |     |              |     |              | 
    ([ 6, 4]  0.012267)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (6, 6)  {
    |              |     ([ 5, 6]  0.017792)     |              | 
    |              |     ([ 6, 6]  0.013016)     |              | 
    |              |     |              |     |              | 
  }
  From:  (6, 7)  {
    |              |     |              |     |              | 
    |              |     ([ 6, 7]  0.010134)     ([ 6, 8]  0.026768) 
    |              |     |              |     |              | 
  }
  From:  (6, 8)  {
    |              |     ([ 5, 8]  0.028831)     |              | 
    |              |     |              |     ([ 6, 9]  0.015182) 
    |              |     |              |     |              | 
  }
  From:  (6, 9)  {
    |              |     |              |     |              | 
    ([ 6, 8]  0.011383)     ([ 6, 9]  0.024369)     ([ 6, 1]  0.011254) 
    |              |     ([ 7, 9]  0.023927)     |              | 
  }
  From:  (7, 1)  {
    |              |     ([ 6, 1]  0.019854)     |              | 
    ([ 7, 9]  0.023181)     |              |     |              | 
    |              |     |              |     |              | 
  }
  From:  (7, 2)  {
    |              |     |              |     |              | 
    ([ 7, 1]  0.024760)     ([ 7, 2]  0.010890)     |              | 
    |              |     |              |     |              | 
  }
  From:  (7, 3)  {
    |              |     ([ 6, 3]  0.017505)     |              | 
    ([ 7, 2]  0.013250)     ([ 7, 3]  0.017844)     ([ 7, 4]  0.025971) 
    |              |     ([ 8, 3]  0.027355)     |              | 
  }
  From:  (7, 4)  {
    |              |     |              |     |              | 
    ([ 7, 3]  0.025090)     ([ 7, 4]  0.011927)     ([ 7, 5]  0.029416) 
    |              |     ([ 8, 4]  0.014680)     |              | 
  }
  From:  (7, 5)  {
    |              |     |              |     |              | 
    |              |     |              |     ([ 7, 6]  0.020943) 
    |              |     ([ 8, 5]  0.018427)     |              | 
  }
  From:  (7, 6)  {
    |              |     ([ 6, 6]  0.020800)     |              | 
    |              |     |              |     ([ 7, 7]  0.025091) 
    |              |     ([ 8, 6]  0.016787)     |              | 
  }
  From:  (7, 7)  {
    |              |     |              |     |              | 
    ([ 7, 6]  0.015619)     |              |     |              | 
    |              |     ([ 8, 7]  0.027751)     |              | 
  }
  From:  (7, 8)  {
    |              |     |              |     |              | 
    ([ 7, 7]  0.023891)     |              |     |              | 
    |              |     ([ 8, 8]  0.026923)     |              | 
  }
  From:  (7, 9)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    ([ 1, 1]  -0.006414)   }
  From:  (8, 1)  {
    |              |     ([ 7, 1]  0.015190)     |              | 
    ([ 8, 9]  0.014230)     ([ 8, 1]  0.015633)     |              | 
    |              |     |              |     |              | 
  }
  From:  (8, 2)  {
    |              |     |              |     |              | 
    |              |     |              |     ([ 8, 3]  0.012990) 
    |              |     ([ 9, 2]  0.018266)     |              | 
  }
  From:  (8, 3)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     ([ 9, 3]  0.016255)     |              | 
  }
  From:  (8, 4)  {
    |              |     |              |     |              | 
    |              |     |              |     |              | 
    |              |     ([ 9, 4]  0.017703)     |              | 
  }
  From:  (8, 5)  {
    |              |     |              |     |              | 
    ([ 8, 4]  0.017633)     |              |     ([ 8, 6]  0.019311) 
    |              |     |              |     |              | 
  }
  From:  (8, 6)  {
    |              |     |              |     |              | 
    |              |     ([ 8, 6]  0.018903)     |              | 
    |              |     ([ 9, 6]  0.016900)     |              | 
  }
  From:  (8, 7)  {
    |              |     ([ 7, 7]  0.017599)     |              | 
    |              |     |              |     |              | 
    |              |     ([ 9, 7]  0.028514)     |              | 
  }
  From:  (8, 8)  {
    |              |     ([ 7, 8]  0.012627)     |              | 
    |              |     |              |     ([ 8, 9]  0.026307) 
    |              |     |              |     |              | 
  }
  From:  (8, 9)  {
    |              |     ([ 7, 9]  0.017152)     |              | 
    ([ 8, 8]  0.012057)     ([ 8, 9]  0.027960)     |              | 
    |              |     |              |     |              | 
  }
  From:  (9, 1)  {
    |              |     |              |     |              | 
    ([ 9, 9]  0.028322)     ([ 9, 1]  0.013561)     ([ 9, 2]  0.017788) 
    |              |     ([ 1, 1]  0.025401)     |              | 
  }
  From:  (9, 2)  {
    |              |     |              |     |              | 
    ([ 9, 1]  0.014937)     ([ 9, 2]  0.022300)     ([ 9, 3]  0.026896) 
    |              |     |              |     |              | 
  }
  From:  (9, 3)  {
    |              |     |              |     |              | 
    |              |     |              |     ([ 9, 4]  0.027075) 
    |              |     |              |     |              | 
  }
  From:  (9, 4)  {
    |              |     |              |     |              | 
    |              |     |              |     ([ 9, 5]  0.016190) 
    |              |     |              |     |              | 
  }
  From:  (9, 5)  {
    |              |     ([ 8, 5]  0.018032)     |              | 
    ([ 9, 4]  0.023579)     |              |     ([ 9, 6]  0.021859) 
    |              |     ([ 1, 5]  0.012789)     |              | 
  }
  From:  (9, 6)  {
    |              |     |              |     |              | 
    |              |     ([ 9, 6]  0.010372)     ([ 9, 7]  0.029489) 
    |              |     |              |     |              | 
  }
  From:  (9, 7)  {
    |              |     ([ 8, 7]  0.015479)     |              | 
    ([ 9, 6]  0.011747)     |              |     |              | 
    |              |     ([ 1, 7]  0.017697)     |              | 
  }
  From:  (9, 8)  {
    |              |     ([ 8, 8]  0.025011)     |              | 
    |              |     |              |     |              | 
    |              |     ([ 1, 8]  0.024583)     |              | 
  }
  From:  (9, 9)  {
    |              |     ([ 8, 9]  0.026073)     |              | 
    |              |     |              |     |              | 
    |              |     |              |     |              | 
  }
}