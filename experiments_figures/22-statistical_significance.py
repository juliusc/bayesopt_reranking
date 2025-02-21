import h5py
import numpy as np
import scipy.stats
import argparse

args = argparse.ArgumentParser()
args.add_argument("group", type=int, default=1, choices=[1, 2, 3, 4])
args = args.parse_args()

data_basic = h5py.File('computed/results-base/all_0.7_1.h5', 'r')
data_bayeslogprob = h5py.File('computed/results-multi/all_0.7_1_avg_logprob_200.h5', 'r')
data_S_50 = h5py.File('computed/results-multi/all_0.7_1_S_50.h5', 'r')
data_S_200 = h5py.File('computed/results-multi/all_0.7_1_S_200.h5', 'r')
data_M_50 = h5py.File('computed/results-multi/all_0.7_1_M_50.h5', 'r')
data_M_200 = h5py.File('computed/results-multi/all_0.7_1_M_200.h5', 'r')


# M_SLICE = 30
def get_slice(data):
    if args.group == 1:
        return data[30]
    if args.group == 2:
        return data[60]
    if args.group == 3:
        return data[90]
    if args.group == 4:
        return data[10:190]

TITLE = {
    1: r"Budget 30 \hspace{2cm}",
    2: r"Budget 60 \hspace{2cm}",
    3: r"Budget 90 \hspace{2cm}",
    4: r"Across budgets 10 to 190 \hspace{-0.2cm}",
}

group_all = {}
group_all["UniqRandom"] = data_basic["random_deduped_scores"]
group_all["LogprobAvg"] = data_basic["avg_logprob_first_scores"]
group_all["LogprobSum"] = data_basic["sum_logprob_first_scores"]
group_all["HillClimbing"] = data_basic["hill_climbing_scores"]
group_all["ProxyFirst 200 Distilled-S"] = data_S_200["proxy_first_scores"]
group_all["ProxyFirst 200 Distilled-M"] = data_M_200["proxy_first_scores"]
group_all["ProxyFirst 50 Distilled-S"] = data_S_50["proxy_first_scores"]
group_all["ProxyFirst 50 Distilled-M"] = data_M_50["proxy_first_scores"]
group_all["hline"] = None
group_all["BayesOpt+GP"] = data_basic["bayesopt_scores"]

group_all["BayesOpt+GP with LogprobAvg"] = data_bayeslogprob["bayesopt_scores"]
group_all["BayesOpt+GP with 200 Distilled-S"] = data_S_200["bayesopt_scores"]
group_all["BayesOpt+GP with 200 Distilled-M"] = data_M_200["bayesopt_scores"]
group_all["BayesOpt+GP with 50 Distilled-S"] = data_S_50["bayesopt_scores"]
group_all["BayesOpt+GP with 50 Distilled-M"] = data_M_50["bayesopt_scores"]

def big_comparer(group1, group2):
    # compares all entries in data1 with all entries in data2 and prints them as a latex table

    # header row
    print(
        r"\bf \Large " + TITLE[args.group]",
        *[r"\rotatebox{90}{" + x + "}" for x in group2.keys() if x != "hline"],
        sep=" & ", end=" \\\\\n"
    )
    print(r"\midrule")

    for i1, (key1, data1) in enumerate(group1.items()):
        # skip first row
        if i1 == 0:
            continue
        if key1 == "hline":
            print(r"\midrule")
            continue
        print(f"{key1:>45}", end=" & ")
        for i2, (key2, data2) in enumerate(group2.items()):
            if key2 == "hline":
                continue

            data1_local_ = get_slice(data1)
            data2_local_ = get_slice(data2)
            # trim if some methods don't have the full data
            data1_local_ = data1_local_[:len(data2_local_)]
            data2_local_ = data2_local_[:len(data1_local_)]
            data1_local_ = np.array(data1_local_).flatten()
            data2_local_ = np.array(data2_local_).flatten()
            data1_local = data1_local_[(data1_local_ > 0.01) & (data2_local_ > 0.01)]
            data2_local = data2_local_[(data1_local_ > 0.01) & (data2_local_ > 0.01)]

            # data1 > data2
            pval1 = scipy.stats.ttest_rel(data1_local, data2_local, alternative="greater")[1]
            # data1 < data2
            pval2 = scipy.stats.ttest_rel(data2_local, data1_local, alternative="greater")[1]

            if pval1 < 0.01:
                out = r"$\leftarrow$"
            elif pval2 < 0.01:
                out = r"$\uparrow$  "
            else:
                out = r"            "
            print(out, end=" & ")
            # print(f"{key1} vs {key2}: {pval1}")
        print("\\\\")
    print(r"\bottomrule")    


big_comparer(
    group_all,
    {k:v for k,v in group_all.items() if "ProxyFirst" not in k and "with" not in k}
)