import os

if not os.path.exists('./results'): os.mkdir('./results')

for i in range(30):
    # os.system('python c5_rankreg.py')
    os.system('python r7_zhangloss.py')
    # os.system('python r6_fangloss.py')

    # os.system('python r1_quantile.py')
    # os.system('python r2_ranknet.py')
    # os.system('python r3_listmle.py')
    # os.system('python r4_listls.py')
