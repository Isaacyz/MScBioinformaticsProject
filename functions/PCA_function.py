import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def PCA_plot(Exp_Dataset,Clinic,colmn):
    if len(set(Clinic[colmn]))>7:
        return print("Too many groups!")
    # Exp_Dataset must be STANDARDIZED
    # Specific the column used for PCA plot
    Exp_Dataset = Exp_Dataset.T
    x = Exp_Dataset.values
    Exp_Dataset = pd.concat([Clinic[colmn], Exp_Dataset],axis=1, join='inner')
    y = Exp_Dataset[colmn].to_list()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['target'] = y

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = set(y)
    all_colors = ['b','g','r','c','m','y','k']
    for target, color in zip(targets, all_colors[0:len(targets)]):
        indicesToKeep = principalDf['target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                   , principalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    plt.xlabel("PC1 {:.3f}".format(pca.explained_variance_ratio_[0]))
    plt.ylabel("PC2 {:.3f}".format(pca.explained_variance_ratio_[1]))
    plt.savefig("a_PCA_plot_{}.png".format(colmn))


