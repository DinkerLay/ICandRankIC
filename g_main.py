import os

if not os.path.exists('./results'): os.mkdir('./results')

for i in range(10):
    # os.system('python g1_lsloss.py')
    # os.system('python g2_listmle.py')
    # os.system('python g3_normmse.py')
    os.system('python g4_aploss.py')
