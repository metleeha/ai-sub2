import sys
import numpy as np

################
# Progress Bar #
################
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s __ %s / %s' % (prefix, bar, percent, '%', suffix, iteration, total))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# Calc Accuracy
def getAcc(Y_pred, Y):
    return (np.array(Y_pred) == np.array(Y)).mean()