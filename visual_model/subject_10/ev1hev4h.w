% Fri Aug 21 23:06:03 2015

% Input layer: (9, 9)
% Output layer: (9, 9)
% Fanout size: (1, 5)
% Fanout spacing: (1, 1)
% Specified fanout weights

Connect(ev1h, ev4h)  {
  From:  (1, 1)  {
    |              |     |              |     |              |     ([ 1, 2]  0.036471)     ([ 1, 3]  0.049909) 
  }
  From:  (1, 2)  {
    ([ 1, 9]  0.042237)     ([ 1, 1]  0.038127)     ([ 1, 2]  0.047136)     ([ 1, 3]  0.036583)     |              | 
  }
  From:  (1, 3)  {
    |              |     ([ 1, 2]  0.048879)     |              |     ([ 1, 4]  0.032899)     ([ 1, 5]  0.044495) 
  }
  From:  (1, 4)  {
    |              |     |              |     |              |     ([ 1, 5]  0.040284)     |              | 
  }
  From:  (1, 5)  {
    |              |     ([ 1, 4]  0.044853)     ([ 1, 5]  0.033041)     ([ 1, 6]  0.049872)     |              | 
  }
  From:  (1, 6)  {
    |              |     |              |     |              |     |              |     |              | 
    ([ 1, 1]  0.039798)   }
  From:  (1, 7)  {
    |              |     |              |     |              |     ([ 1, 8]  0.037731)     |              | 
  }
  From:  (1, 8)  {
    ([ 1, 6]  0.030476)     ([ 1, 7]  0.047682)     ([ 1, 8]  0.039387)     |              |     ([ 1, 1]  0.041285) 
  }
  From:  (1, 9)  {
    ([ 1, 7]  0.044833)     |              |     |              |     |              |     ([ 1, 2]  0.043049) 
  }
  From:  (2, 1)  {
    |              |     ([ 2, 9]  0.044200)     ([ 2, 1]  0.049010)     ([ 2, 2]  0.037706)     |              | 
  }
  From:  (2, 2)  {
    ([ 2, 9]  0.033957)     ([ 2, 1]  0.039125)     |              |     |              |     |              | 
  }
  From:  (2, 3)  {
    ([ 2, 1]  0.048638)     |              |     ([ 2, 3]  0.043526)     ([ 2, 4]  0.043399)     ([ 2, 5]  0.035109) 
  }
  From:  (2, 4)  {
    |              |     |              |     |              |     ([ 2, 5]  0.030048)     ([ 2, 6]  0.039504) 
  }
  From:  (2, 5)  {
    |              |     ([ 2, 4]  0.045754)     |              |     |              |     ([ 2, 7]  0.034164) 
  }
  From:  (2, 6)  {
    ([ 2, 4]  0.043616)     ([ 2, 5]  0.045898)     |              |     ([ 2, 7]  0.031156)     |              | 
  }
  From:  (2, 7)  {
    ([ 2, 5]  0.043713)     ([ 2, 6]  0.042313)     ([ 2, 7]  0.032975)     ([ 2, 8]  0.038642)     |              | 
  }
  From:  (2, 8)  {
    ([ 2, 6]  0.031781)     ([ 2, 7]  0.033960)     ([ 2, 8]  0.037081)     |              |     ([ 2, 1]  0.035538) 
  }
  From:  (2, 9)  {
    |              |     |              |     ([ 2, 9]  0.033205)     |              |     |              | 
  }
  From:  (3, 1)  {
    |              |     ([ 3, 9]  0.049530)     ([ 3, 1]  0.034911)     |              |     |              | 
  }
  From:  (3, 2)  {
    ([ 3, 9]  0.032189)     ([ 3, 1]  0.044018)     ([ 3, 2]  0.030738)     |              |     ([ 3, 4]  0.041124) 
  }
  From:  (3, 3)  {
    |              |     ([ 3, 2]  0.036818)     ([ 3, 3]  0.038018)     |              |     ([ 3, 5]  0.047082) 
  }
  From:  (3, 4)  {
    |              |     |              |     |              |     |              |     ([ 3, 6]  0.037257) 
  }
  From:  (3, 5)  {
    |              |     ([ 3, 4]  0.031928)     ([ 3, 5]  0.036502)     |              |     ([ 3, 7]  0.048447) 
  }
  From:  (3, 6)  {
    |              |     |              |     |              |     |              |     |              | 
    ([ 1, 1]  0.044033)   }
  From:  (3, 7)  {
    |              |     ([ 3, 6]  0.046455)     ([ 3, 7]  0.042994)     ([ 3, 8]  0.044938)     ([ 3, 9]  0.034230) 
  }
  From:  (3, 8)  {
    ([ 3, 6]  0.042245)     ([ 3, 7]  0.034594)     |              |     ([ 3, 9]  0.039500)     ([ 3, 1]  0.036888) 
  }
  From:  (3, 9)  {
    ([ 3, 7]  0.043970)     |              |     ([ 3, 9]  0.037371)     |              |     |              | 
  }
  From:  (4, 1)  {
    ([ 4, 8]  0.048058)     |              |     ([ 4, 1]  0.033794)     |              |     ([ 4, 3]  0.046649) 
  }
  From:  (4, 2)  {
    ([ 4, 9]  0.046144)     |              |     |              |     |              |     ([ 4, 4]  0.048745) 
  }
  From:  (4, 3)  {
    |              |     |              |     |              |     |              |     ([ 4, 5]  0.045273) 
  }
  From:  (4, 4)  {
    ([ 4, 2]  0.042031)     |              |     ([ 4, 4]  0.038091)     |              |     |              | 
  }
  From:  (4, 5)  {
    |              |     |              |     |              |     ([ 4, 6]  0.039431)     ([ 4, 7]  0.041659) 
  }
  From:  (4, 6)  {
    |              |     |              |     ([ 4, 6]  0.035509)     ([ 4, 7]  0.045764)     ([ 4, 8]  0.041365) 
  }
  From:  (4, 7)  {
    |              |     |              |     ([ 4, 7]  0.045270)     |              |     |              | 
  }
  From:  (4, 8)  {
    |              |     |              |     ([ 4, 8]  0.043062)     ([ 4, 9]  0.031648)     ([ 4, 1]  0.048944) 
  }
  From:  (4, 9)  {
    ([ 4, 7]  0.045075)     ([ 4, 8]  0.036021)     ([ 4, 9]  0.043857)     ([ 4, 1]  0.038202)     |              | 
  }
  From:  (5, 1)  {
    |              |     ([ 5, 9]  0.036748)     ([ 5, 1]  0.031785)     ([ 5, 2]  0.038216)     ([ 5, 3]  0.037135) 
  }
  From:  (5, 2)  {
    ([ 5, 9]  0.049408)     |              |     ([ 5, 2]  0.034920)     ([ 5, 3]  0.038061)     |              | 
  }
  From:  (5, 3)  {
    |              |     ([ 5, 2]  0.042579)     ([ 5, 3]  0.041575)     ([ 5, 4]  0.042864)     |              | 
  }
  From:  (5, 4)  {
    |              |     ([ 5, 3]  0.035143)     |              |     |              |     ([ 5, 6]  0.031125) 
  }
  From:  (5, 5)  {
    |              |     |              |     ([ 5, 5]  0.040328)     |              |     |              | 
  }
  From:  (5, 6)  {
    ([ 5, 4]  0.045097)     ([ 5, 5]  0.040155)     ([ 5, 6]  0.049656)     |              |     |              | 
  }
  From:  (5, 7)  {
    ([ 5, 5]  0.047257)     |              |     ([ 5, 7]  0.037833)     ([ 5, 8]  0.033899)     |              | 
  }
  From:  (5, 8)  {
    ([ 5, 6]  0.044048)     |              |     |              |     |              |     ([ 5, 1]  0.042531) 
  }
  From:  (5, 9)  {
    ([ 5, 7]  0.037766)     ([ 5, 8]  0.034383)     ([ 5, 9]  0.045329)     ([ 5, 1]  0.041049)     ([ 5, 2]  0.031697) 
  }
  From:  (6, 1)  {
    ([ 6, 8]  0.031895)     ([ 6, 9]  0.042344)     ([ 6, 1]  0.048851)     |              |     ([ 6, 3]  0.041897) 
  }
  From:  (6, 2)  {
    |              |     ([ 6, 1]  0.036281)     ([ 6, 2]  0.036473)     ([ 6, 3]  0.047352)     ([ 6, 4]  0.032133) 
  }
  From:  (6, 3)  {
    ([ 6, 1]  0.040779)     |              |     |              |     ([ 6, 4]  0.034831)     |              | 
  }
  From:  (6, 4)  {
    |              |     |              |     |              |     |              |     |              | 
    ([ 1, 1]  0.040155)   }
  From:  (6, 5)  {
    ([ 6, 3]  0.044236)     ([ 6, 4]  0.036098)     |              |     ([ 6, 6]  0.048361)     |              | 
  }
  From:  (6, 6)  {
    |              |     |              |     |              |     ([ 6, 7]  0.039575)     ([ 6, 8]  0.040824) 
  }
  From:  (6, 7)  {
    |              |     ([ 6, 6]  0.037962)     ([ 6, 7]  0.047981)     ([ 6, 8]  0.042615)     ([ 6, 9]  0.034169) 
  }
  From:  (6, 8)  {
    ([ 6, 6]  0.045074)     |              |     ([ 6, 8]  0.039467)     ([ 6, 9]  0.047046)     ([ 6, 1]  0.046083) 
  }
  From:  (6, 9)  {
    |              |     |              |     ([ 6, 9]  0.035907)     |              |     ([ 6, 2]  0.035219) 
  }
  From:  (7, 1)  {
    |              |     |              |     |              |     ([ 7, 2]  0.034786)     ([ 7, 3]  0.034281) 
  }
  From:  (7, 2)  {
    |              |     |              |     ([ 7, 2]  0.048074)     ([ 7, 3]  0.048610)     ([ 7, 4]  0.046953) 
  }
  From:  (7, 3)  {
    |              |     ([ 7, 2]  0.030325)     |              |     |              |     |              | 
  }
  From:  (7, 4)  {
    ([ 7, 2]  0.041201)     |              |     ([ 7, 4]  0.040625)     ([ 7, 5]  0.048717)     |              | 
  }
  From:  (7, 5)  {
    ([ 7, 3]  0.045441)     ([ 7, 4]  0.038368)     |              |     ([ 7, 6]  0.031243)     ([ 7, 7]  0.049190) 
  }
  From:  (7, 6)  {
    |              |     ([ 7, 5]  0.039880)     |              |     |              |     |              | 
  }
  From:  (7, 7)  {
    |              |     |              |     |              |     ([ 7, 8]  0.043385)     |              | 
  }
  From:  (7, 8)  {
    ([ 7, 6]  0.049585)     |              |     |              |     |              |     ([ 7, 1]  0.034882) 
  }
  From:  (7, 9)  {
    ([ 7, 7]  0.040685)     |              |     |              |     |              |     ([ 7, 2]  0.035618) 
  }
  From:  (8, 1)  {
    |              |     |              |     |              |     |              |     ([ 8, 3]  0.047397) 
  }
  From:  (8, 2)  {
    |              |     |              |     ([ 8, 2]  0.038512)     ([ 8, 3]  0.035904)     ([ 8, 4]  0.039597) 
  }
  From:  (8, 3)  {
    |              |     ([ 8, 2]  0.042970)     |              |     |              |     |              | 
  }
  From:  (8, 4)  {
    |              |     ([ 8, 3]  0.043607)     |              |     ([ 8, 5]  0.034880)     ([ 8, 6]  0.031872) 
  }
  From:  (8, 5)  {
    ([ 8, 3]  0.033906)     ([ 8, 4]  0.037806)     |              |     ([ 8, 6]  0.046076)     ([ 8, 7]  0.048534) 
  }
  From:  (8, 6)  {
    ([ 8, 4]  0.045919)     ([ 8, 5]  0.045210)     |              |     |              |     |              | 
  }
  From:  (8, 7)  {
    ([ 8, 5]  0.045520)     ([ 8, 6]  0.035601)     |              |     |              |     ([ 8, 9]  0.033131) 
  }
  From:  (8, 8)  {
    ([ 8, 6]  0.049255)     |              |     ([ 8, 8]  0.049877)     |              |     |              | 
  }
  From:  (8, 9)  {
    |              |     ([ 8, 8]  0.042157)     |              |     ([ 8, 1]  0.038656)     |              | 
  }
  From:  (9, 1)  {
    |              |     ([ 9, 9]  0.049923)     ([ 9, 1]  0.048369)     ([ 9, 2]  0.033981)     |              | 
  }
  From:  (9, 2)  {
    ([ 9, 9]  0.043786)     ([ 9, 1]  0.036384)     ([ 9, 2]  0.046030)     ([ 9, 3]  0.037780)     ([ 9, 4]  0.037826) 
  }
  From:  (9, 3)  {
    |              |     |              |     ([ 9, 3]  0.047770)     |              |     ([ 9, 5]  0.037430) 
  }
  From:  (9, 4)  {
    ([ 9, 2]  0.031565)     ([ 9, 3]  0.042095)     ([ 9, 4]  0.037653)     ([ 9, 5]  0.047423)     ([ 9, 6]  0.033036) 
  }
  From:  (9, 5)  {
    ([ 9, 3]  0.042845)     |              |     |              |     ([ 9, 6]  0.040148)     |              | 
  }
  From:  (9, 6)  {
    ([ 9, 4]  0.038039)     ([ 9, 5]  0.030526)     ([ 9, 6]  0.035567)     |              |     ([ 9, 8]  0.038767) 
  }
  From:  (9, 7)  {
    ([ 9, 5]  0.032551)     ([ 9, 6]  0.036300)     ([ 9, 7]  0.034499)     |              |     ([ 9, 9]  0.040091) 
  }
  From:  (9, 8)  {
    ([ 9, 6]  0.032141)     ([ 9, 7]  0.038146)     ([ 9, 8]  0.048960)     |              |     ([ 9, 1]  0.032574) 
  }
  From:  (9, 9)  {
    |              |     ([ 9, 8]  0.038742)     ([ 9, 9]  0.047489)     |              |     ([ 9, 2]  0.038188) 
  }
}