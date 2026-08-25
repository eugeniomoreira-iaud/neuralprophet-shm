import re

tex_path = "studies/02_proxy_forcing_characterization/report/proxy_forcing_report.tex"
with open(tex_path, "r") as f:
    content = f.read()

# Fix table 10 wrapper
table10_wrong = r"\\input\{../outputs/PF_T10_tair_agreement\.tex\}"
table10_fixed = r"""\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{llrrrrrrrrr}
\\toprule
\\textbf{Source} & \\textbf{Season} & \\textbf{N} & \\textbf{Bias} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{r} & \\textbf{Amp (On)} & \\textbf{Amp (Prx)} & \\textbf{Amp Ratio} & \\textbf{Phase Diff} \\\\
\\midrule
\\input{../outputs/PF_T10_tair_agreement.tex}
\\bottomrule
\\end{tabular}
\\caption{Air-temperature agreement metrics.}
\\label{tab:tair_agreement}
\\end{table}"""
content = content.replace(table10_wrong, table10_fixed)

# Fix table 11 wrapper
table11_wrong = r"\\input\{../outputs/PF_T11_sr_agreement\.tex\}"
table11_fixed = r"""\\begin{table}[H]
\\centering
\\small
\\begin{tabular}{llrrrrrrrrr}
\\toprule
\\textbf{Source} & \\textbf{Season} & \\textbf{N} & \\textbf{Bias} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{r} & \\textbf{Amp (On)} & \\textbf{Amp (Prx)} & \\textbf{Amp Ratio} & \\textbf{Phase Diff} \\\\
\\midrule
\\input{../outputs/PF_T11_sr_agreement.tex}
\\bottomrule
\\end{tabular}
\\caption{Solar-radiation agreement metrics.}
\\label{tab:sr_agreement}
\\end{table}"""
content = content.replace(table11_wrong, table11_fixed)

with open(tex_path, "w") as f:
    f.write(content)
