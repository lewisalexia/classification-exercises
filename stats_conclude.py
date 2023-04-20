# STATS CONCLUSIONS FUNCTIONS 


def chi2_test(table):
    α = 0.05
    chi2, pval, degf, expected = stats.chi2_contingency(table)
    print('Observed')
    print(table.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p-value = {pval:.4f}')
    print('----')
    if pval < α:
        print ('We reject the null hypothesis.')
    else:
        print ("We fail to reject the null hypothesis.")





def conclude_1samp_tt(group1, group_mean):
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if ((p < α) & (tstat > 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_1samp_gt(group1, group_mean):
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if ((p / 2) < α) and (tstat > 0):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')





def conclude_1samp_lt(group1, group_mean):
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if ((p / 2) < α) and (tstat < 0):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_2samp_tt(sample1, sample2):
    α = 0.05
    stat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if p < α:
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_2samp_gt(sample1, sample2):
    α = 0.05
    stat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if (((p/2) < α) and (tstat > 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_2samp_lt(sample1, sample2):
    α = 0.05
    stat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if (((p/2) < α) and (tstat < 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')





def conclude_anova(theoretical_mean, group1, group2):
    α = 0.05
    tstat, pval = stats.f_oneway(theoretical_mean, group1, group2)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(pval)
    print('----')
    if pval < α:
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_pearsonr(floats1, floats2):
    α = 0.05
    r, p = stats.pearsonr(floats1, floats2)
    print('r =', r)
    print('p =', p)
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")




def conclude_spearmanr(floats1, floats2):
    α = 0.05
    r, p = stats.spearmanr(floats1, floats2)
    print('r =', r)
    print('p =', p)
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")




def conclude_mannwhitneyu(subpop1, subpop2):
    α = 0.05
    print(f"$H_0$: {subpop1} is independent of {subpop2}")
    print()
    print(f"$H_a$: {subpop1} is dependent of {subpop2}")
    t, p = stats.mannwhitneyu(subpop1, subpop2)
    print(f'P-Value = ')
    p
    if p < α:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")