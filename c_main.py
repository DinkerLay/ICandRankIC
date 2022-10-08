import os

if not os.path.exists('./results'): os.mkdir('./results')

for i in range(30):
    # os.system('python c1_reg.py')
    # os.system('python c4_transreg.py')
    os.system('python c2_icreg.py')
    os.system('python c3_scalereg.py')
    # os.system('python r1_quantile.py')
    # os.system('python r2_ranknet.py')
    # os.system('python r3_listmle.py')
    # os.system('python r4_listls.py')
