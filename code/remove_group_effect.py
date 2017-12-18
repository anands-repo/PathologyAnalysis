import scipy.stats
import numpy as np
import scipy.linalg
import pandas
from statsmodels.formula.api import ols

def mask_group_effect(expression_data, phenotypes, fexclude=0.7):
    """
    expression_data is [# patients x # marker genes]
    phenotypes is [# patients]
    fexclude is the fraction of genes that will be discarded

    Returns a mask matrix to mask genes that are highly correlated with phenotype
    """
    """ >>>>> First, we determine which genes correlate with phenotype <<<<< """

    expression_array = np.split(expression_data.T, expression_data.T.shape[0]);
    pvalue_array     = [];

    """ Find linear-fit p-values for each gene """
    for expr in expression_array:
        x = expr.flatten();
        y = phenotypes;
        data = pandas.DataFrame({'x':x, 'y':y});
        model = ols("x ~ y", data).fit();
        pvalue_array.append(model.f_pvalue)

    pvalue_array = np.array(pvalue_array);

    """ Find the (1-fexclude) lowest p-value subset """
    percentile_cutoff = np.percentile(pvalue_array, fexclude);
    percentile_mask   = np.where(pvalue_array > percentile_cutoff, np.zeros(percentile_cutoff.shape), np.ones(percentile_cutoff.shape));
    W = np.diag(percentile_mask);

    """ Create masked expression data """
    expression_masked = np.matmul(W, expression_data.T);

    """ Determine surrogate population variables """
    U, A, Vt     = scipy.linalg.svd(expression_masked); 
    s_init       = Vt.T[:,0];
    pvalue_array = [];

    """ >>>>> Next, we determine which of the components of s_init have a group effect <<<<< """
    for expr in expression_array:
        x = expr.flatten();
        y = s_init;
        z = phenotypes;
        w = (phenotypes * s_init);
        data = pandas.DataFrame({'x':x, 'y':y, 'z':z, 'w':w});
        model = ols("x ~ y + z + w", data).fit();

        pvalue_array.append(model.f_pvalue)

    pvalue_array = np.array(pvalue_array);

    """ Find the (1-fexclude)**2 lowest p-value subset """
    percentile_cutoff = np.percentile(pvalue_array, 1 - (1 - fexclude)**2);
    percentile_mask   = np.where(pvalue_array > percentile_cutoff, np.zeros(percentile_cutoff.shape), np.ones(percentile_cutoff.shape));
    W = np.diag(percentile_mask);

    return W;
