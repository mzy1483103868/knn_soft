import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

def K_cla(file_adress,test_data):
    data = pd.read_csv(file_adress)
    y = data["severity"].copy()
    T = data[["birth_year", "area", "a", "b", "c"]].copy()
    X = data[["birth_year", "area", "severity"]].copy()
    z = data["area"].copy()
    clors = sns.color_palette('hls', 40)
#    sns.palplot(clors)
#    plt.figure(figsize=(5,5))
#    X_test = np.array([[1911, 15, 2, 1, 3], [1915, 33, 1, 2, 1]])
    X_test = np.array(test_data)
    are11 = X.loc[z == 11]
    are12 = X.loc[z == 12]
    are13 = X.loc[z == 13]
    are14 = X.loc[z == 14]
    are15 = X.loc[z == 15]
    are21 = X.loc[z == 21]
    are22 = X.loc[z == 22]
    are23 = X.loc[z == 23]
    are31 = X.loc[z == 31]
    are32 = X.loc[z == 32]
    are33 = X.loc[z == 33]
    are34 = X.loc[z == 34]
    are35 = X.loc[z == 35]
    are36 = X.loc[z == 36]
    are37 = X.loc[z == 37]
    are41 = X.loc[z == 41]
    are42 = X.loc[z == 42]
    are43 = X.loc[z == 43]
    are44 = X.loc[z == 44]
    are45 = X.loc[z == 45]
    are46 = X.loc[z == 46]
    are50 = X.loc[z == 50]
    are51 = X.loc[z == 51]
    are52 = X.loc[z == 52]
    are53 = X.loc[z == 53]
    are54 = X.loc[z == 54]
    are61 = X.loc[z == 61]
    are62 = X.loc[z == 62]
    are63 = X.loc[z == 63]
    are64 = X.loc[z == 64]
    are65 = X.loc[z == 65]
    are71 = X.loc[z == 71]
    are81 = X.loc[z == 81]
    are82 = X.loc[z == 82]
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(T, y)
    X_result = clf.predict(X_test)
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.scatter(are11['birth_year'], are11['area'], are11['severity'], s=100, c=clors[0])
    ax1.scatter(are12['birth_year'], are12['area'], are12['severity'], s=100, c=clors[1])
    ax1.scatter(are13['birth_year'], are13['area'], are13['severity'], s=100, c=clors[2])
    ax1.scatter(are14['birth_year'], are14['area'], are14['severity'], s=100, c=clors[3])
    ax1.scatter(are15['birth_year'], are15['area'], are15['severity'], s=100, c=clors[4])
    ax1.scatter(are21['birth_year'], are21['area'], are21['severity'], s=100, c=clors[5])
    ax1.scatter(are22['birth_year'], are22['area'], are22['severity'], s=100, c=clors[6])
    ax1.scatter(are23['birth_year'], are23['area'], are23['severity'], s=100, c=clors[7])
    ax1.scatter(are31['birth_year'], are31['area'], are31['severity'], s=100, c=clors[8])
    ax1.scatter(are32['birth_year'], are32['area'], are32['severity'], s=100, c=clors[9])
    ax1.scatter(are33['birth_year'], are33['area'], are33['severity'], s=100, c=clors[10])
    ax1.scatter(are34['birth_year'], are34['area'], are34['severity'], s=100, c=clors[11])
    ax1.scatter(are35['birth_year'], are35['area'], are35['severity'], s=100, c=clors[12])
    ax1.scatter(are36['birth_year'], are36['area'], are36['severity'], s=100, c=clors[13])
    ax1.scatter(are37['birth_year'], are37['area'], are37['severity'], s=100, c=clors[14])
    ax1.scatter(are41['birth_year'], are41['area'], are41['severity'], s=100, c=clors[15])
    ax1.scatter(are42['birth_year'], are42['area'], are42['severity'], s=100, c=clors[16])
    ax1.scatter(are43['birth_year'], are43['area'], are43['severity'], s=100, c=clors[17])
    ax1.scatter(are44['birth_year'], are44['area'], are44['severity'], s=100, c=clors[18])
    ax1.scatter(are45['birth_year'], are45['area'], are45['severity'], s=100, c=clors[19])
    ax1.scatter(are46['birth_year'], are46['area'], are46['severity'], s=100, c=clors[20])
    ax1.scatter(are50['birth_year'], are50['area'], are50['severity'], s=100, c=clors[21])
    ax1.scatter(are51['birth_year'], are51['area'], are51['severity'], s=100, c=clors[22])
    ax1.scatter(are52['birth_year'], are52['area'], are52['severity'], s=100, c=clors[23])
    ax1.scatter(are53['birth_year'], are53['area'], are53['severity'], s=100, c=clors[24])
    ax1.scatter(are54['birth_year'], are54['area'], are54['severity'], s=100, c=clors[25])
    ax1.scatter(are61['birth_year'], are61['area'], are61['severity'], s=100, c=clors[26])
    ax1.scatter(are62['birth_year'], are62['area'], are62['severity'], s=100, c=clors[27])
    ax1.scatter(are63['birth_year'], are63['area'], are63['severity'], s=100, c=clors[28])
    ax1.scatter(are64['birth_year'], are64['area'], are64['severity'], s=100, c=clors[29])
    ax1.scatter(are65['birth_year'], are65['area'], are65['severity'], s=100, c=clors[30])
    ax1.scatter(are71['birth_year'], are71['area'], are71['severity'], s=100, c=clors[31])
    ax1.scatter(are81['birth_year'], are81['area'], are81['severity'], s=100, c=clors[32])
    ax1.scatter(are82['birth_year'], are82['area'], are82['severity'], s=100, c=clors[33])
    ax1.set_xlabel("birth")
    ax1.set_ylabel("area")
    ax1.set_zlabel("Severity")
    for i in range(0, 1):
        X_result_c = clors[39]
        if test_data[i][1] == 11:
            X_result_c = clors[0]
        elif test_data[i][1] == 12:
            X_result_c = clors[1]
        elif test_data[i][1] == 13:
            X_result_c = clors[2]
        elif test_data[i][1] == 14:
            X_result_c = clors[3]
        elif test_data[i][1] == 15:
            X_result_c = clors[4]
        elif test_data[i][1] == 21:
            X_result_c = clors[5]
        elif test_data[i][1] == 22:
            X_result_c = clors[6]
        elif test_data[i][1] == 23:
            X_result_c = clors[7]
        elif test_data[i][1] == 31:
            X_result_c = clors[8]
        elif test_data[i][1] == 32:
            X_result_c = clors[9]
        elif test_data[i][1] == 33:
            X_result_c = clors[10]
        elif test_data[i][1] == 34:
            X_result_c = clors[11]
        elif test_data[i][1] == 35:
            X_result_c = clors[12]
        elif test_data[i][1] == 36:
            X_result_c = clors[13]
        elif test_data[i][1] == 37:
            X_result_c = clors[14]
        elif test_data[i][1] == 41:
            X_result_c = clors[15]
        elif test_data[i][1] == 42:
            X_result_c = clors[16]
        elif test_data[i][1] == 43:
            X_result_c = clors[17]
        elif test_data[i][1] == 44:
            X_result_c = clors[18]
        elif test_data[i][1] == 45:
            X_result_c = clors[19]
        elif test_data[i][1] == 46:
            X_result_c = clors[20]
        elif test_data[i][1] == 50:
            X_result_c = clors[21]
        elif test_data[i][1] == 51:
            X_result_c = clors[22]
        elif test_data[i][1] == 52:
            X_result_c = clors[23]
        elif test_data[i][1] == 53:
            X_result_c = clors[24]
        elif test_data[i][1] == 54:
            X_result_c = clors[25]
        elif test_data[i][1] == 61:
            X_result_c = clors[26]
        elif test_data[i][1] == 62:
            X_result_c = clors[27]
        elif test_data[i][1] == 63:
            X_result_c = clors[28]
        elif test_data[i][1] == 64:
            X_result_c = clors[29]
        elif test_data[i][1] == 65:
            X_result_c = clors[30]
        elif test_data[i][1] == 71:
            X_result_c = clors[31]
        elif test_data[i][1] == 81:
            X_result_c = clors[32]
        elif test_data[i][1] == 82:
            X_result_c = clors[33]
        ax1.scatter(X_test[i, 0], X_test[i, 1], X_result[i], s=100, marker='*', c=X_result_c)
    plt.show()



'''
if __name__ == '__main__':
#    adress="testdata123.csv"
    adress="file:///Users/meizhangyu/Desktop/KNN_soft/testdata123.csv"
#    testdata=[[1911, 15, 2, 1, 3], [1915, 33, 1, 2, 1]]
    testdata=[[1911, 15, 2, 1, 3]]
    number=1
    K_cla(adress,testdata)
'''