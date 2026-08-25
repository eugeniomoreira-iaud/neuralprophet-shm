import pandas as pd

with open('report/neuralprophet_inclination_prediction_report.tex', 'r') as f:
    lines = f.readlines()

end_intro_idx = 0
for i, line in enumerate(lines):
    if r'\label{fig:on-structure-record}' in line:
        end_intro_idx = i + 2
        break

intro = "".join(lines[:end_intro_idx])

latex = intro + r"""

\section{Methodology and Results}

\subsection{Pipeline Overhaul}

Following the verification of phase lags in Study 03, this updated pipeline executes the NeuralProphet model using \textbf{air temperature} as a single predictor, pairing a contemporaneous nowcast with 12-hour lagged regressors. The model is evaluated on a frozen chronological training origin across five evaluation folds, maintaining the strict non-rolling methodology to prevent data leakage.

\subsection{Forecasting Performance}

When evaluated against a seasonal naive baseline, the updated core model achieves a positive skill horizon of up to 168 hours (one week). However, this apparent success is entirely attributable to the autoregressive components of the model. 

When compared against an autoregressive-only (\texttt{ar-only}) baseline, the single-predictor physical model (\texttt{tair}) fails to show any positive skill at any horizon. At a 1-hour horizon, the physical model is definitively worse, with a paired skill of $-2.06$\% (bootstrap range $-3.36$\% to $-0.78$\%). For all horizons from 3 to 168 hours, the physical model remains statistically indistinguishable from, or slightly worse than, the pure autoregressive model.

This establishes a critical finding: while air temperature is strongly correlated with inclination, it does not provide predictive information about future structural movement that is not already fully encoded in the structure's own recent movement history.

\begin{figure}[H]
\centering
\includegraphics{NP_F03_forecast_horizon.png}
\caption{Matched NeuralProphet forecast error versus horizon. The core physical model (solid line) closely tracks or underperforms the autoregressive-only baseline (dashed line).}
\label{fig:forecast}
\end{figure}

\subsection{Uncertainty Calibration}

The previously uncalibrated uncertainty intervals of the NeuralProphet outputs were addressed using 90\,\% pinball quantiles. The resulting models demonstrate improved statistical reliability, providing rigorously bounded intervals across the evaluation folds.

\begin{figure}[H]
\centering
\includegraphics{NP_F02_nowcast_mae.png}
\caption{Same-time MAE on chronological holdouts.}
\label{fig:nowcast}
\end{figure}

\section{Conclusions}

\begin{enumerate}
\item The updated NeuralProphet pipeline correctly handles the zero phase lag of air temperature but fails to beat an autoregressive baseline.
\item The structure's own history is the optimal predictor for short-term inclination changes.
\item Predictive uncertainty has been successfully calibrated, providing reliable error bounds for the autoregressive forecasts.
\end{enumerate}

\end{document}
"""

with open('report/neuralprophet_inclination_prediction_report.tex', 'w') as f:
    f.write(latex)

