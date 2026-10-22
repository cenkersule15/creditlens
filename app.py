import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
import streamlit as st

from credit_risk.data import BILLS, FEATURES, PAYMENTS, ROOT, STATUS, TARGET, validate_features
from credit_risk.evaluation import cost_at_threshold, threshold_curve
from credit_risk.model import calibrated_probability, explain, load_bundle
from credit_risk.monitoring import drift_report
from credit_risk.ui import CSS, TURKISH

st.set_page_config(page_title="CreditLens | Explainable Risk", page_icon="◈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def resources():
    return load_bundle()


@st.cache_data
def data():
    folder = ROOT / "artifacts"
    return (json.loads((folder / "report.json").read_text()), pd.read_csv(folder / "test_predictions.csv"), pd.read_csv(folder / "demo_customers.csv"), json.loads((folder / "reference_profile.json").read_text()))


def chart(figure, height=340):
    palette = ["#5dd6c0", "#8c9fff", "#f4b860", "#ee7c87"]
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color="#b9c9df", size=12), margin=dict(l=20, r=22, t=40, b=42), height=height, colorway=palette, legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font_size=11), hoverlabel=dict(bgcolor="#203249", bordercolor="#466078", font_size=12), bargap=.16, barcornerradius=4)
    figure.update_xaxes(gridcolor="#26354b", griddash="dot", zerolinecolor="#3b4e67", tickfont_size=11, title_font_size=12, automargin=True)
    figure.update_yaxes(gridcolor="#26354b", griddash="dot", zerolinecolor="#3b4e67", tickfont_size=11, title_font_size=12, automargin=True)
    if any(trace.type == "bar" and trace.orientation == "h" for trace in figure.data):
        figure.update_yaxes(showgrid=False)
    for index, trace in enumerate(figure.data):
        if trace.type == "scatter":
            trace.line.width = 2.4
        if trace.type in ("bar", "histogram") and trace.name not in [t("Risiko steigt", "Risk increases"), t("Risiko sinkt", "Risk decreases")]:
            trace.marker.color = palette[index % len(palette)]
        if trace.name:
            trace.name = tr(trace.name)
    for axis in [figure.layout.xaxis, figure.layout.yaxis]:
        if axis.title.text:
            axis.title.text = tr(axis.title.text)
    if language == "Türkçe":
        for trace in figure.data:
            for dimension in ("x", "y"):
                values = getattr(trace, dimension, None)
                if values is not None and len(values) and isinstance(values[0], str):
                    setattr(trace, dimension, [tr(value) for value in values])
    st.plotly_chart(figure, width="stretch")


if not (ROOT / "artifacts/model.joblib").exists():
    st.error("Model artifacts are missing. Run: python -m credit_risk.train")
    st.stop()

bundle = resources()
report, test, demos, profile = data()
st.sidebar.markdown('<div class="sidebar-brand"><span class="brand-mark"><svg viewBox="0 0 28 28" fill="none" aria-hidden="true"><path d="M4 19V9l10-5 10 5v10l-10 5-10-5Z" stroke="currentColor" stroke-width="1.6"/><path d="m8 17 4-5 4 3 5-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span><div><div class="brand-name">CreditLens</div><div class="brand-desc">EXPLAINABLE CREDIT RISK</div></div></div>', unsafe_allow_html=True)
language = st.sidebar.selectbox("Sprache / Language / Dil", ["Deutsch", "English", "Türkçe"], key="interface_language")
de = language == "Deutsch"


def tr(english):
    return TURKISH.get(english, english) if language == "Türkçe" else english


def display_table(frame):
    if language != "Türkçe":
        return frame
    labels = {"roc_auc": "ROC-AUC", "average_precision": "Ortalama kesinlik (AP)", "brier_score": "Brier skoru", "log_loss": "Log kaybı", "precision": "Kesinlik", "recall": "Duyarlılık", "rows": "Satır sayısı", "default_rate": "Temerrüt oranı", "attribute": "Öznitelik", "group": "Grup", "n": "Müşteri sayısı", "defaults": "Temerrüt sayısı", "mean_prediction": "Ortalama tahmin", "review_rate": "İnceleme oranı", "true_positive_rate": "Doğru pozitif oranı", "false_positive_rate": "Yanlış pozitif oranı"}
    result = frame.copy()
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(lambda value: TURKISH.get(value, value) if isinstance(value, str) else value)
    return result.rename(columns=labels, index={"train": "Eğitim", "calibration": "Kalibrasyon", "validation": "Doğrulama", "test": "Test"})


def t(german, english):
    return german if de else tr(english)



pages = [t("Portfolioübersicht", "Portfolio overview"), t("Einzelfallanalyse", "Customer analysis"), t("Risikosimulation", "Risk simulation"), t("Datenmonitoring", "Data monitoring"), t("Methodik & Modell", "Methodology & model")]
page = st.sidebar.radio(t("Arbeitsbereich", "Workspace"), pages, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Cenker Sule · AI & Data Science")
st.sidebar.caption(t("Forschungsdemo · Taiwan, 2005\nBeträge in NT$, nicht EUR.", "Research demo · Taiwan, 2005\nAmounts in NT$, not EUR."))
edition = t("FORSCHUNGSDEMO", "Research edition")
st.markdown(f'<div class="brandline"><span class="eyebrow">CREDITLENS / RISK INTELLIGENCE</span><span class="brand-chip">{edition}</span></div>', unsafe_allow_html=True)
st.title(page)

if page == pages[0]:
    st.markdown(f'<p class="subtle">{t("Ausfallrisiken verstehen. Prognosen erklären. Modellqualität messen.", "Understand default risk. Explain predictions. Measure model quality.")}</p>', unsafe_allow_html=True)
    m = report["test_metrics"]["XGBoost calibrated"]
    cols = st.columns(4)
    for col, label, value in zip(cols, [t("Testkunden", "Test customers"), "ROC-AUC", tr("Average Precision"), tr("Brier Score ↓")], [f"{len(test):,}", f"{m['roc_auc']:.3f}", f"{m['average_precision']:.3f}", f"{m['brier_score']:.3f}"]):
        col.metric(label, value)
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader(t("Risikoverteilung", "Risk distribution"))
        plot = test.assign(outcome=test[TARGET].map({0: t("Kein Ausfall", "No default"), 1: t("Ausfall", "Default")}))
        chart(px.histogram(plot, x="probability", color="outcome", nbins=30, barmode="overlay", opacity=.75, labels={"probability": t("Geschätzte Ausfallwahrscheinlichkeit", "Estimated default probability"), "outcome": ""}))
    with right:
        st.subheader(t("Wichtigste Modellfaktoren", "Leading model factors"))
        imp = pd.read_csv(ROOT / "artifacts/importance.csv").head(8).sort_values("mean_abs_shap")
        chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", labels={"mean_abs_shap": "Mean |SHAP| · calibrated log-odds", "feature": ""}))
    st.caption(t("Kennzahlen aus dem getrennten Testdatensatz. SHAP-Rangfolge aus 300 Trainingsprofilen; Zusammenhänge sind keine Kausalität.", "Metrics use the separate test partition. SHAP ranking uses 300 training profiles; associations are not causation."))
    a, b = st.columns(2)
    with a:
        st.subheader("ROC")
        fig = go.Figure()
        for name, col in [("XGBoost calibrated", "probability"), ("Logistic Regression", "logistic_probability")]:
            fpr, tpr, _ = roc_curve(test[TARGET], test[col])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dot")))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        chart(fig)
    with b:
        st.subheader(tr("Precision–Recall"))
        fig = go.Figure()
        for name, col in [("XGBoost calibrated", "probability"), ("Logistic Regression", "logistic_probability")]:
            precision, recall, _ = precision_recall_curve(test[TARGET], test[col])
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))
        fig.add_hline(y=test[TARGET].mean(), line_dash="dot")
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision")
        chart(fig)

elif page == pages[1]:
    st.caption(t("Bestehende Kreditkartenkunden · sechs Monate Zahlungshistorie · Beträge in NT$", "Existing credit-card customers · six-month payment history · amounts in NT$"))
    choice = st.selectbox(t("Beispielprofil", "Example profile"), range(3), format_func=lambda i: t(["Niedrige Modellprognose", "Mittlere Modellprognose", "Hohe Modellprognose"][i], ["Low model prediction", "Middle model prediction", "High model prediction"][i]))
    row = demos.iloc[[choice]][FEATURES].copy()
    st.caption(t("Monat 1 = September 2005, Monat 6 = April 2005. Statuscodes werden als Originalwerte beibehalten.", "Month 1 = September 2005, month 6 = April 2005. Repayment codes retain their original values."))
    limit = st.number_input(t("Kreditlimit (NT$)", "Credit limit (NT$)"), min_value=1, value=int(row.LIMIT_BAL.iloc[0]), step=1000, key=f"limit_{choice}")
    history = pd.DataFrame({"Month": ["Sep", "Aug", "Jul", "Jun", "May", "Apr"], "Status": row[STATUS].iloc[0].values.astype(int), "Bill (NT$)": row[BILLS].iloc[0].values.astype(int), "Paid (NT$)": row[PAYMENTS].iloc[0].values.astype(int)})
    edited = st.data_editor(history, hide_index=True, disabled=["Month"], width="stretch", key=f"history_{choice}", column_config={"Month": st.column_config.TextColumn(label=tr("Month")), "Bill (NT$)": st.column_config.NumberColumn(label=tr("Bill (NT$)")), "Status": st.column_config.NumberColumn(label=tr("Status"), min_value=-2, max_value=9, step=1), "Paid (NT$)": st.column_config.NumberColumn(label=tr("Paid (NT$)"), min_value=0, step=100)})
    row.loc[:, "LIMIT_BAL"] = limit
    row.loc[:, STATUS] = edited["Status"].to_numpy()
    row.loc[:, BILLS] = edited["Bill (NT$)"].to_numpy()
    row.loc[:, PAYMENTS] = edited["Paid (NT$)"].to_numpy()
    try:
        factors, base, probability = explain(bundle, row)
        a, b, c = st.columns(3)
        a.metric(t("Ausfallwahrscheinlichkeit", "Default probability"), f"{probability:.1%}")
        b.metric(t("Validierungsschwelle", "Validation threshold"), f"{bundle['threshold']:.0%}")
        c.metric(t("Prüfhinweis", "Review signal"), t("Prüfen", "Review") if probability >= bundle["threshold"] else t("Unter Schwelle", "Below threshold"))
        factors = factors.assign(absolute=factors.contribution.abs()).nlargest(10, "absolute").sort_values("contribution")
        factors["direction"] = np.where(factors.contribution > 0, t("Risiko steigt", "Risk increases"), t("Risiko sinkt", "Risk decreases"))
        chart(px.bar(factors, x="contribution", y="feature", color="direction", orientation="h", color_discrete_map={t("Risiko steigt", "Risk increases"): "#ee7c87", t("Risiko sinkt", "Risk decreases"): "#5dd6c0"}, labels={"contribution": t("SHAP-Beitrag zu kalibrierten Log-Odds", "SHAP contribution to calibrated log-odds"), "feature": "", "direction": ""}), 420)
        st.caption(t("Alle 19 Beiträge plus Basiswert ergeben die kalibrierten Log-Odds; die Grafik zeigt die 10 größten Beiträge. Keine Prozentpunkte und keine kausalen Effekte.", "All 19 contributions plus the base value reconstruct calibrated log-odds; the chart shows the 10 largest contributions. These are not percentage points or causal effects."))
        outside = drift_report(profile, validate_features(row))
        if outside.outside_training_range.max() > 0:
            st.warning(t("Mindestens ein Wert liegt außerhalb des Trainingsbereichs. Die Prognose kann unzuverlässig sein.", "At least one value is outside the training range. The prediction may be unreliable."))
    except ValueError as error:
        st.error(tr(str(error)))

elif page == pages[2]:
    st.caption(t("Validierungsdaten für die Simulation. Der Testdatensatz bleibt für die feste Referenzbewertung reserviert.", "Simulation uses validation data. Test results remain a fixed reference evaluation."))
    col1, col2, col3 = st.columns(3)
    missed_cost = col1.slider(t("Kosten je übersehenem Ausfall", "Cost per missed default"), 1, 20, 5)
    review_cost = col2.slider(t("Kosten je Prüfung", "Cost per review"), 1, 5, 1)
    threshold = col3.slider(t("Risikoschwelle", "Risk threshold"), 0.0, 1.0, float(bundle["threshold"]), .01)
    original = pd.read_csv(ROOT / "artifacts/validation_predictions.csv")
    curve = threshold_curve(original[TARGET], original.probability, missed_cost, review_cost)
    policy = cost_at_threshold(original[TARGET], original.probability, threshold, missed_cost, review_cost)
    best = curve.loc[curve.cost_per_customer.idxmin()]
    a, b, c, d = st.columns(4)
    a.metric(t("Prüfquote", "Review rate"), f"{policy['review_rate']:.1%}")
    b.metric(t("Übersehene Ausfälle", "Missed defaults"), policy["missed_defaults"])
    c.metric(t("Kosten je Kunde", "Cost per customer"), f"{policy['cost_per_customer']:.3f}")
    d.metric(t("Beste Gitterschwelle", "Best grid threshold"), f"{best.threshold:.0%}")
    fig = px.line(curve, x="threshold", y="cost_per_customer", labels={"threshold": t("Risikoschwelle", "Risk threshold"), "cost_per_customer": t("Illustrative Kosteneinheiten / Kunde", "Illustrative cost units / customer")})
    fig.add_vline(x=threshold, line_color="#f4b860", line_dash="dash")
    chart(fig, 420)
    st.info(t("Annahme: Jede markierte Person wird geprüft; markierte Ausfälle verursachen danach keine weiteren Ausfallkosten. Prüfungen verhindern reale Verluste nicht automatisch. Dies sind illustrative Einheiten, keine EUR-Beträge oder tatsächlichen Einsparungen.", "Assumption: every flagged customer is reviewed; flagged defaults incur no additional default cost afterwards. Reviews do not automatically prevent real losses. These are illustrative units, not EUR amounts or actual savings."))

elif page == pages[3]:
    st.caption(t("CSV mit 19 Modellmerkmalen hochladen. Zusätzliche Spalten werden für Prognosen ignoriert.", "Upload a CSV containing the 19 model features. Extra columns are ignored for predictions."))
    st.download_button(t("Beispiel-CSV herunterladen", "Download example CSV"), (ROOT / "data/batch_example.csv").read_bytes(), "batch_example.csv", "text/csv")
    upload = st.file_uploader(tr("Upload customer CSV"), type="csv")
    if upload:
        try:
            batch = validate_features(pd.read_csv(upload))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(tr(str(error)))
            st.stop()
    else:
        scenario = st.radio(t("Demoszenario", "Demo scenario"), [t("Testportfolio", "Test portfolio"), t("Verschobene Kreditlimits", "Shifted credit limits")], horizontal=True)
        batch = test[FEATURES].copy()
        if scenario != t("Testportfolio", "Test portfolio"):
            batch["LIMIT_BAL"] *= 2
            st.caption(t("Künstliche Verschiebung zur Demonstration; keine beobachtete zeitliche Veränderung.", "Artificial shift for demonstration; not observed temporal drift."))
    monitoring = drift_report(profile, batch)
    a, b, c = st.columns(3)
    a.metric(t("Datensätze", "Rows"), len(batch))
    b.metric(t("Maximaler PSI", "Maximum PSI"), f"{monitoring.psi.max():.3f}")
    c.metric(t("Merkmale mit PSI ≥ 0,25", "Features with PSI ≥ 0.25"), int((monitoring.psi >= .25).sum()))
    chart(px.bar(monitoring.head(10).sort_values("psi"), x="psi", y="feature", orientation="h", labels={"feature": "", "psi": "Population Stability Index"}))
    display_monitoring = monitoring.copy()
    if language == "Türkçe":
        display_monitoring["signal"] = display_monitoring["signal"].map(tr)
        display_monitoring = display_monitoring.rename(columns={"feature": "Özellik", "outside_training_range": "Eğitim aralığı dışındaki oran", "signal": "Sinyal"})
    st.dataframe(display_monitoring, hide_index=True, width="stretch")
    st.caption(t("PSI nutzt Trainingsquantile. 0,10/0,25 sind heuristische Hinweise, keine Signifikanztests. Kleine Stichproben und Werte innerhalb eines Bins können irreführen. Ohne Labels misst PSI keine Modellgüte.", "PSI uses training quantiles. 0.10/0.25 are heuristic signals, not significance tests. Small samples and shifts within a bin can mislead. Without labels, PSI does not measure model quality."))
    predicted = batch.copy()
    predicted["default_probability"] = calibrated_probability(bundle, batch)
    predicted["review_flag"] = predicted.default_probability >= bundle["threshold"]
    st.download_button(t("Prognosen herunterladen", "Download predictions"), predicted.to_csv(index=False).encode(), "predictions.csv", "text/csv")

else:
    st.subheader(t("Modellvergleich · getrennte Testdaten", "Model comparison · held-out test data"))
    comparison = pd.DataFrame(report["test_metrics"]).T.drop(columns="confusion_matrix")
    st.dataframe(display_table(comparison).style.format("{:.4f}"), width="stretch")
    a, b = st.columns(2)
    with a:
        st.subheader(t("Kalibrierung", "Calibration"))
        fig = go.Figure()
        for name, column in [("Raw", "raw_probability"), ("Calibrated", "probability")]:
            observed, predicted = calibration_curve(test[TARGET], test[column], n_bins=10, strategy="quantile")
            fig.add_trace(go.Scatter(x=predicted, y=observed, name=name, mode="lines+markers"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Ideal", line=dict(dash="dot")))
        fig.update_xaxes(title="Mean predicted probability")
        fig.update_yaxes(title="Observed default rate")
        chart(fig)
    with b:
        st.subheader(t("Konfusionsmatrix", "Confusion matrix"))
        chart(px.imshow(report["test_metrics"]["XGBoost calibrated"]["confusion_matrix"], x=["Not flagged", "Flagged"], y=["No default", "Default"], text_auto=True, color_continuous_scale="Teal", labels={"x": "Prediction", "y": "Observed"}))
    intervals = report["confidence_intervals_95"]
    st.caption(tr("95% bootstrap intervals · ") + " · ".join(f"{key}: {value[0]:.3f}–{value[1]:.3f}" for key, value in intervals.items()))
    st.subheader(t("Aufteilung & Nachvollziehbarkeit", "Partitions & traceability"))
    st.dataframe(display_table(pd.DataFrame(report["splits"]).T), width="stretch")
    st.write(t("Finanziell identische Profile bleiben in derselben Partition. Training, Kalibrierung, Validierung und Test sind getrennt. Drei XGBoost-Konfigurationen werden anhand der Average Precision auf der Validierung verglichen. Der Schwellenwert wird ebenfalls auf der Validierung gewählt.", "Financially identical profiles remain in the same partition. Training, calibration, validation and test are separate. Three XGBoost configurations are compared using validation Average Precision. The threshold is also selected on validation."))
    st.subheader(t("Subgruppenprüfung", "Subgroup audit"))
    st.dataframe(display_table(pd.DataFrame(report["subgroups"])), hide_index=True, width="stretch")
    st.caption(t("Geschlecht und Alter dienen nur der Auswertung. Ihr Ausschluss aus dem Modell beweist keine Fairness; Stellvertretereffekte und Unsicherheit bleiben möglich.", "Sex and age are used only for evaluation. Excluding them from model inputs does not establish fairness; proxy effects and uncertainty remain possible."))
    st.info(t("Historische Taiwan-Daten von 2005, keine aktuelle deutsche Bankpopulation. Keine zeitliche Validierung oder Produktionsfreigabe. Ziel ist der Zahlungsausfall bestehender Kartenkunden im Folgemonat, keine automatische Kreditvergabe.", "Historical Taiwan data from 2005, not a current German banking population. No temporal validation or production approval. The target is next-month default for existing cardholders, not automated lending."))
    st.markdown("[UCI dataset · Yeh (2009) · CC BY 4.0](https://doi.org/10.24432/C55S3H)")
