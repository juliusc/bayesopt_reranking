import h5py
data_basic = h5py.File('computed/results-base/all_0.7_1.h5', 'r')
data_bayeslogprob = h5py.File('computed/results-multi/all_0.7_1_avg_logprob_200.h5', 'r')["bayesopt_score"]
data_S_50 = h5py.File('computed/results-multi/all_0.7_1_S_50.h5', 'r')
data_S_100 = h5py.File('computed/results-multi/all_0.7_1_S_100.h5', 'r')
data_S_200 = h5py.File('computed/results-multi/all_0.7_1_S_200.h5', 'r')
data_M_50 = h5py.File('computed/results-multi/all_0.7_1_M_50.h5', 'r')
data_M_100 = h5py.File('computed/results-multi/all_0.7_1_M_100.h5', 'r')
data_M_200 = h5py.File('computed/results-multi/all_0.7_1_M_200.h5', 'r')

def plot_row(name, places, data):
    # 10, 40, 70, 100, 130, 160, 190
    data_sub = data[10::30]
    print(
        name,
        places,
        *[f"{x:.4f}" for x in data_sub],
        sep=" & ",
        end=" \\\\\n"
    )

print(r"\toprule")
print(
    r"",
    r"",
    r"\multicolumn{6}{c}{\bf CometKiwi runs}",
    sep=" & ",
    end=" \\\\\n",
)
print(
    r"\bf Method",
    r"\bf Figure",
    *[f"\\bf {x}" for x in range(10, 200, 30)],
    sep=" & ",
    end=" \\\\\n",
)
print(r"\midrule")

plot_row("UniqRandom", r"\ref{fig:results_baselines}", data_basic["random_deduped_score"])
plot_row("LogprobAvg", r"\ref{fig:results_baselines}", data_basic["avg_logprob_first_score"])
plot_row("LogprobSum", r"\ref{fig:results_baselines}", data_basic["sum_logprob_first_score"])
plot_row("HillClimbing", r"\ref{fig:results_baselines}", data_basic["hill_climbing_score"])
plot_row("ProxyFirst 200 Distilled-S", r"\ref{fig:results_multi_SM}", data_S_200["proxy_first_score"])
plot_row("ProxyFirst 200 Distilled-M", r"\ref{fig:results_multi_SM}", data_M_200["proxy_first_score"])
plot_row("ProxyFirst 50 Distilled-S", r"\ref{fig:results_multi_SM}", data_S_50["proxy_first_score"])
plot_row("ProxyFirst 50 Distilled-M", r"\ref{fig:results_multi_SM}", data_M_50["proxy_first_score"])
print(r"\cmidrule{1-1}")
plot_row("BayesOpt+GP", r"\ref{fig:results_baselines},\ref{fig:results_multi},\ref{fig:results_multi_SM}", data_basic["bayesopt_score"])
plot_row("BayesOpt+GP with LogprobAvg", r"\ref{fig:results_multi}", data_bayeslogprob)
plot_row("BayesOpt+GP with 200 Distilled-S", r"\ref{fig:results_multi},\ref{fig:results_multi_SM}", data_S_200["bayesopt_score"])
plot_row("BayesOpt+GP with 200 Distilled-M", r"\ref{fig:results_multi},\ref{fig:results_multi_SM}", data_M_200["bayesopt_score"])
plot_row("BayesOpt+GP with 50 Distilled-S", r"\ref{fig:results_multi},\ref{fig:results_multi_SM}", data_S_50["bayesopt_score"])
plot_row("BayesOpt+GP with 50 Distilled-M", r"\ref{fig:results_multi},\ref{fig:results_multi_SM}", data_M_50["bayesopt_score"])


print(r"\bottomrule")