"""
Interactive dashboard for the CXL memory tiering study.

Run from anywhere:
    streamlit run dashboard/app.py

The previous version resolved its data with "../results/...", which only
worked if you happened to launch it from inside dashboard/ -- and the
README told you to launch it from the repo root, so it always failed.
Paths are now resolved relative to this file.

There is also a zero-install version of this dashboard at
dashboard/index.html: open it in any browser, no Python required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

st.set_page_config(
    page_title="Expert Tiering Console",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Palette ------------------------------------------------------------
# Shared with dashboard/index.html so the two tell the same visual story.
S = {
    "static": "#2a78d6",    # blue
    "lru": "#eb6834",       # orange
    "periodic": "#1baf7a",  # aqua
    "hybrid": "#eda100",    # yellow
    "prefetch": "#e87ba4",  # magenta
    "muted": "#898781",
}
LABEL = {
    "static": "Pure Static", "lru": "Pure LRU", "periodic": "Periodic re-profile",
    "hybrid": "Hybrid", "prefetch": "Markov prefetch",
}

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1400px; }
  h1, h2, h3 { letter-spacing: -0.015em; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
  [data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: .1em;
      font-size: .68rem; opacity: .75; }
  .caveat { border-left: 3px solid #fab219; padding: .35rem 0 .35rem .9rem;
      font-size: .88rem; opacity: .88; margin: .6rem 0; }
</style>
""", unsafe_allow_html=True)


# --- Shared timing model (mirrors src/tier_simulator.py) -----------------
def access_time_ns(latency_ns, bandwidth_gbps, size_bytes):
    return latency_ns + (size_bytes / 1024 ** 3) / bandwidth_gbps * 1e9


def fmt_time(ns):
    return f"{ns / 1000:,.1f} us" if ns >= 1000 else f"{ns:,.0f} ns"


@st.cache_data
def load(name):
    """Load one results CSV, or None if that stage has not been run yet."""
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def plotly_theme(fig, height=380, ylab=None, xlab=None):
    fig.update_layout(
        height=height, template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, title=xlab, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", title=ylab, zeroline=False)
    return fig


tiering = load("tiering_sweep_results.csv")
three = load("three_strategy_comparison.csv")
reprofile = load("reprofile_interval_sweep.csv")
energy = load("energy_metrics.csv")
phases = load("hybrid_nonstationary_comparison.csv")
best_phase = load("reprofile_best_by_phase.csv")
robust = load("robustness_check_summary.csv")
prefetch = load("prefetch_multi_dataset.csv")
real_hits = load("real_trace_hit_rates.csv")

if tiering is None:
    st.error(
        f"No results found in {RESULTS}.\n\n"
        "Generate them first:  `python run_all.py`"
    )
    st.stop()


# ======================================================================== #
# Sidebar: the hardware model every page reads from
# ======================================================================== #
with st.sidebar:
    st.markdown("### Memory configuration")
    st.caption("Every latency on this dashboard is recomputed from these.")

    hbm_lat = st.slider("HBM latency (ns)", 60, 300, 150, 5)
    hbm_bw = st.slider("HBM bandwidth (GB/s)", 200, 1600, 800, 25)
    cxl_add = st.slider("CXL added latency (ns)", 20, 400, 70, 5)
    cxl_bw = st.slider("CXL bandwidth (GB/s)", 16, 256, 64, 4)

    size_label = st.select_slider(
        "Expert size",
        options=["16 MiB", "32 MiB", "64 MiB", "128 MiB", "336 MiB (Mixtral fp16)", "672 MiB"],
        value="16 MiB",
    )
    size_mib = int(size_label.split()[0])
    expert_bytes = size_mib * 1024 ** 2

    t_hbm = access_time_ns(hbm_lat, hbm_bw, expert_bytes)
    t_cxl = access_time_ns(hbm_lat + cxl_add, cxl_bw, expert_bytes)

    st.divider()
    st.metric("HBM access", fmt_time(t_hbm))
    st.metric("CXL access", fmt_time(t_cxl))
    st.metric("Miss penalty", f"{t_cxl / t_hbm:.1f}x")

    if size_mib != 16:
        st.markdown(
            '<div class="caveat">The committed results use 16 MiB. '
            'Changing this rescales every latency but leaves hit rates -- and '
            'therefore the policy ranking -- untouched.</div>',
            unsafe_allow_html=True)

    st.divider()
    st.caption("Zero-install version: `dashboard/index.html`")


def mix(hit_pct):
    """Average access time at a given HBM hit rate, under the sidebar config."""
    h = hit_pct / 100.0
    return h * t_hbm + (1 - h) * t_cxl


# ======================================================================== #
st.title("Expert tiering across HBM and CXL")
st.markdown(
    "Mixture-of-Experts models keep eight experts per layer resident but touch only "
    "two per token. This measures what happens when the cold ones are demoted to "
    "CXL-attached memory, and which placement policy keeps the hot ones local."
)

if real_hits is not None:
    hits = real_hits[~real_hits["Strategy"].str.contains("ORACLE")]
    l15 = hits[hits["Layer"] == "Layer 15"].set_index("Strategy")["HBM Hit Rate (%)"]
    l31 = hits[hits["Layer"] == "Layer 31"].set_index("Strategy")["HBM Hit Rate (%)"]
    source_note = "measured, held-out profiling"
else:
    l15 = pd.Series({"Pure Static": 57.51, "Periodic Re-profile": 61.02, "Pure LRU": 64.94})
    l31 = pd.Series({"Pure Static": 65.76, "Pure LRU": 65.91, "Periodic Re-profile": 67.26})
    source_note = "published values -- run `run_all.py --stage real-trace` to refresh"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Layer 15 best", f"{l15.max():.1f}%",
          f"+{l15.max() - l15.min():.1f} pts vs worst")
c2.metric("Layer 31 best", f"{l31.max():.1f}%",
          f"+{l31.max() - l31.min():.1f} pts vs worst")
if energy is not None:
    e15 = energy[energy["Layer"] == "Layer 15"]["Total Energy (mJ)"]
    c3.metric("Memory energy saved", f"{(1 - e15.min() / e15.max()) * 100:.1f}%",
              "layer 15, best vs worst policy")
c4.metric("Capacity freed at k=4", f"{4 * size_mib:,} MiB", "50% of expert weights")
st.caption(f"Hit rates: {source_note}.")

st.divider()

tabs = st.tabs([
    "Capacity vs latency",
    "Placement intelligence",
    "Depth changes the winner",
    "Shifting workloads",
    "Re-profiling cadence",
    "Energy",
    "Prefetching",
    "Robustness",
    "Method & limits",
])

# ------------------------------------------------------------------ #
with tabs[0]:
    st.subheader("What capacity costs in latency")
    st.latex(r"t_{access} = L_{tier} + \frac{S_{expert}}{B_{tier}}")
    st.markdown(
        "Access **counts** below are measured on the trace; only the timing model "
        "comes from the sidebar. Move the sliders and the curves move with them."
    )

    k = st.slider("Experts resident in HBM", 0, 8, 4, key="cap_k")
    row = tiering[tiering["num_experts_in_hbm"] == k].iloc[0]
    hit_static = row["hbm_accesses"] / row["total_accesses"] * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg access time", fmt_time(mix(hit_static)))
    m2.metric("HBM hit rate", f"{hit_static:.1f}%")
    m3.metric("Capacity freed", f"{(8 - k) * size_mib:,} MiB", f"{(8 - k) * 12.5:.1f}% of weights")
    m4.metric("vs HBM-only", f"{mix(hit_static) / t_hbm:.2f}x", "slower")

    ks = tiering["num_experts_in_hbm"].to_numpy()
    static_hits = tiering["hbm_accesses"] / tiering["total_accesses"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[mix(h) for h in static_hits], name="Static (frequency-ranked)",
                             line=dict(color=S["static"], width=3), mode="lines+markers"))
    if three is not None:
        fig.add_trace(go.Scatter(x=ks, y=[mix(h) for h in three["lru_hit_rate_pct"]], name="LRU (recency)",
                                 line=dict(color=S["lru"], width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ks, y=[mix(kk / 8 * 100) for kk in ks], name="Random placement",
                             line=dict(color=S["muted"], width=2, dash="dash"), mode="lines+markers"))
    fig.add_vline(x=k, line=dict(color="rgba(128,128,128,.5)", width=1, dash="dot"))
    st.plotly_chart(plotly_theme(fig, ylab="Avg access time (ns)",
                                 xlab="Experts resident in HBM"), width="stretch")

    st.markdown(
        '<div class="caveat"><b>Read the slope, not the intercept.</b> The model charges a '
        'full expert transfer on every access, so absolute times sit far above a real decode '
        'step, where a resident expert is read at tile granularity and never re-fetched. How '
        'much each demoted expert costs is the result that survives.</div>',
        unsafe_allow_html=True)

# ------------------------------------------------------------------ #
with tabs[1]:
    st.subheader("Does ranking experts actually pay?")
    st.markdown(
        '"Less CXL is faster" is true whichever experts you pick, so it proves nothing. '
        "The honest test fixes the capacity budget and asks whether a hot/cold ranking "
        "beats an arbitrary one -- here, against 20 random placements per budget."
    )
    svr = load("smart_vs_random_results.csv")
    if svr is None:
        st.info("Run `python run_all.py --stage baseline` to generate this.")
    else:
        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=svr["num_experts_in_hbm"], y=svr["random_avg_time_ns_mean"],
                error_y=dict(type="data", array=svr["random_avg_time_ns_std"], thickness=1),
                name="Random (20 draws, +/-1 sd)", mode="lines+markers",
                line=dict(color=S["muted"], width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=svr["num_experts_in_hbm"], y=svr["smart_avg_time_ns"],
                name="Frequency-ranked", mode="lines+markers",
                line=dict(color=S["static"], width=3)))
            st.plotly_chart(plotly_theme(fig, ylab="Avg access time (ns)",
                                         xlab="HBM budget"), width="stretch")
        with right:
            fig = go.Figure(go.Bar(
                x=svr["num_experts_in_hbm"], y=svr["smart_advantage_pct"],
                marker_color=S["periodic"],
                hovertemplate="budget %{x}: %{y:.2f}% faster<extra></extra>"))
            st.plotly_chart(plotly_theme(fig, ylab="Faster than random (%)",
                                         xlab="HBM budget"), width="stretch")

        peak = svr.loc[svr["smart_advantage_pct"].idxmax()]
        st.success(
            f"Ranking peaks at **{peak['smart_advantage_pct']:.2f}% faster** with "
            f"{int(peak['num_experts_in_hbm'])} of 8 experts in HBM. "
            "The zeros at budgets 0 and 8 are degenerate -- every placement is the same "
            "placement there -- and are a sanity check on the harness, not a result."
        )

# ------------------------------------------------------------------ #
with tabs[2]:
    st.subheader("The layer decides the policy")
    st.markdown(
        "Same three policies, two depths of the same model, 829,441 recorded routing "
        "decisions each. The ordering inverts."
    )

    order = ["Pure Static", "Periodic Re-profile", "Pure LRU"]
    key = {"Pure Static": "static", "Periodic Re-profile": "periodic", "Pure LRU": "lru"}

    fig = go.Figure()
    for strat in order:
        fig.add_trace(go.Scatter(
            x=[l15.get(strat, np.nan), l31.get(strat, np.nan)], y=[strat, strat],
            mode="lines", line=dict(color="rgba(128,128,128,.45)", width=3),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[l15.get(s, np.nan) for s in order], y=order, mode="markers", name="Layer 15 (middle)",
        marker=dict(size=15, color=[S[key[s]] for s in order])))
    fig.add_trace(go.Scatter(
        x=[l31.get(s, np.nan) for s in order], y=order, mode="markers", name="Layer 31 (deep)",
        marker=dict(size=15, color="rgba(0,0,0,0)",
                    line=dict(width=3, color=[S[key[s]] for s in order]))))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(plotly_theme(fig, height=300, xlab="HBM hit rate (%)"), width="stretch")

    comp = pd.DataFrame({
        "Policy": order,
        "Layer 15": [l15.get(s, np.nan) for s in order],
        "Layer 31": [l31.get(s, np.nan) for s in order],
    })
    comp["Deep - mid"] = comp["Layer 31"] - comp["Layer 15"]
    st.dataframe(comp.style.format({"Layer 15": "{:.2f}%", "Layer 31": "{:.2f}%",
                                    "Deep - mid": "{:+.2f}"}),
                 hide_index=True, width="stretch")

    a, b = st.columns(2)
    a.info(f"**Layer 15 -> {l15.idxmax()}** ({l15.max():.2f}%). Middle layers re-route with "
           "the subject matter, so a frozen top-4 goes stale within a few thousand tokens "
           "and reactivity pays.")
    b.info(f"**Layer 31 -> {l31.idxmax()}** ({l31.max():.2f}%). Deep experts specialise and "
           "stay put; static and LRU finish within "
           f"{abs(l31.get('Pure LRU', 0) - l31.get('Pure Static', 0)):.2f} points of each "
           "other, so adapting per access buys almost nothing.")

# ------------------------------------------------------------------ #
with tabs[3]:
    st.subheader("When the hot set rotates underneath you")
    st.markdown(
        "A three-phase synthetic trace whose hot experts rotate at each boundary. "
        "Static gets the deal a real deployment gets: profile once on phase 0, then live "
        "with it. No policy sees the future."
    )
    if phases is None:
        st.info("Run `python run_all.py --stage hybrid reprofile` to generate this.")
    else:
        series = {
            "static": phases["avg_time_ns_static"],
            "lru": phases["avg_time_ns_lru"],
            "hybrid": phases["avg_time_ns_hybrid"],
        }
        if best_phase is not None:
            series["periodic"] = best_phase["avg_time_ns"]

        fig = go.Figure()
        for k_, vals in series.items():
            fig.add_trace(go.Scatter(x=phases["phase"], y=vals, name=LABEL[k_],
                                     mode="lines+markers", line=dict(color=S[k_], width=3),
                                     marker=dict(size=10)))
        fig.update_xaxes(tickmode="array", tickvals=phases["phase"],
                         ticktext=[f"Phase {p}" for p in phases["phase"]])
        st.plotly_chart(plotly_theme(fig, ylab="Avg access time (ns)",
                                     xlab="Hot experts rotate at each boundary"),
                        width="stretch")

        st_s = phases["avg_time_ns_static"]
        degrade = (st_s.iloc[-1] / st_s.iloc[0] - 1) * 100
        st.warning(
            f"**Static's collapse is the whole story.** It is fastest in phase 0 -- it was "
            f"tuned on phase 0 -- and slowest by phase 2, degrading **{degrade:.1f}%** as the "
            "distribution rotates away from it. Periodic re-profiling is never worst in any "
            "phase, which is what makes it the best overall choice despite never winning "
            "phase 0."
        )

# ------------------------------------------------------------------ #
with tabs[4]:
    st.subheader("How often should you re-profile?")
    st.markdown(
        "Re-ranking is not free -- every re-rank implies migrating experts across the link. "
        "Too frequent and you chase noise in a short window; too rare and you run a stale map."
    )
    if reprofile is None:
        st.info("Run `python run_all.py --stage reprofile` to generate this.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=reprofile["reprofile_interval"],
                                 y=reprofile["overall_avg_time_ns"],
                                 name="Periodic re-profile", mode="lines+markers",
                                 line=dict(color=S["periodic"], width=3), marker=dict(size=10)))
        if phases is not None:
            for k_, col in [("static", "avg_time_ns_static"), ("lru", "avg_time_ns_lru"),
                            ("hybrid", "avg_time_ns_hybrid")]:
                fig.add_hline(y=phases[col].mean(), line=dict(color=S[k_], width=2, dash="dash"),
                              annotation_text=LABEL[k_], annotation_position="right")
        fig.update_xaxes(type="log")
        st.plotly_chart(plotly_theme(fig, ylab="Overall avg access time (ns)",
                                     xlab="Tokens between re-ranks (log)"),
                        width="stretch")

        best = reprofile.loc[reprofile["overall_avg_time_ns"].idxmin()]
        worst = reprofile.loc[reprofile["overall_avg_time_ns"].idxmax()]
        st.success(
            f"Floor at **every {int(best['reprofile_interval']):,} tokens**. Stretched to "
            f"{int(worst['reprofile_interval']):,} it is worse than doing nothing -- it has "
            "become static placement with extra migration traffic. The window must be short "
            "enough to catch a phase change and long enough to estimate a ranking from."
        )

# ------------------------------------------------------------------ #
with tabs[5]:
    st.subheader("Energy follows the hit rate")
    if energy is None:
        st.info("Run `python run_all.py --stage energy` to generate this.")
    else:
        pick = st.radio("Layer", sorted(energy["Layer"].unique()), horizontal=True)
        sub = energy[energy["Layer"] == pick].sort_values("Total Energy (mJ)")
        inv = {"Pure Static": "static", "Pure LRU": "lru", "Periodic Re-profile": "periodic"}
        colors = [S.get(inv.get(s, "muted"), S["muted"]) for s in sub["Strategy"]]

        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure(go.Bar(
                x=sub["Strategy"], y=sub["Total Energy (mJ)"], marker_color=colors,
                text=[f"{v:.2f} mJ" for v in sub["Total Energy (mJ)"]], textposition="outside",
                hovertemplate="%{x}: %{y:.2f} mJ<extra></extra>"))
            fig.update_yaxes(range=[0, sub["Total Energy (mJ)"].max() * 1.2])
            st.plotly_chart(plotly_theme(fig, ylab="Total energy (mJ)"), width="stretch")
        with right:
            show = sub.copy()
            if "Saving vs worst (%)" not in show:
                show["Saving vs worst (%)"] = (1 - show["Total Energy (mJ)"]
                                               / show["Total Energy (mJ)"].max()) * 100
            st.dataframe(
                show[["Strategy", "HBM Hit Rate (%)", "Total Energy (mJ)", "Saving vs worst (%)"]]
                .style.format({"HBM Hit Rate (%)": "{:.2f}%", "Total Energy (mJ)": "{:.2f}",
                               "Saving vs worst (%)": "{:.2f}%"}),
                hide_index=True, width="stretch")

        st.markdown(
            '<div class="caveat"><b>The ratio is sound; the absolute is not.</b> The '
            '21.08 nJ/fetch constant is DRAMSim3 total energy over request count, so it '
            'carries background and refresh power amortised over the run -- not the marginal '
            'cost of one 64 B burst. Every percentage here is a ratio of two figures built '
            'from that same constant, so it cancels and the savings hold. The millijoule '
            'column does not transfer to silicon.</div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ #
with tabs[6]:
    st.subheader("Markov prefetching: a negative result")
    if prefetch is None:
        st.info("Run `python run_all.py --stage prefetch` to generate this.")
    else:
        st.markdown(
            "A prefetch that guesses wrong still crosses the link. Once wasted prefetches are "
            "charged for, the trade-off becomes visible: hit rate against bandwidth and energy."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Runs evaluated", f"{len(prefetch)}")
        m2.metric("Mean prefetch accuracy", f"{prefetch['Prefetch Accuracy (%)'].mean():.1f}%")
        net = prefetch[(prefetch["Hit Rate Gain (pts)"] > 0.5)
                       & (prefetch["Energy Multiplier"] < 1.0)]
        m3.metric("Gained hit rate AND cut energy", f"{len(net)} of {len(prefetch)}")

        fig = go.Figure()
        for gate, grp in prefetch.groupby("Gating"):
            fig.add_trace(go.Scatter(
                x=grp["Energy Multiplier"], y=grp["Hit Rate Gain (pts)"], mode="markers",
                name=gate, marker=dict(size=11, opacity=.85),
                text=grp["Dataset"] + " (order " + grp["Markov Order"].astype(str) + ")",
                hovertemplate="%{text}<br>energy %{x:.2f}x, gain %{y:+.2f} pts<extra></extra>"))
        fig.add_vline(x=1.0, line=dict(color=S["muted"], width=2, dash="dash"),
                      annotation_text="energy break-even")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(plotly_theme(fig, ylab="Hit rate gain (pts)",
                                     xlab="Memory energy vs LRU-only (x)"),
                        width="stretch")

        st.error(
            "**Every point that gains hit rate sits to the right of break-even.** The only "
            "settings that move the hit rate materially fire on nearly every access at around "
            "15% accuracy, and because a wasted prefetch still crosses the link, energy rises "
            "faster than the hit rate does. The gain reported before wasted prefetches were "
            "charged was not a real saving."
        )
        with st.expander("Full results across all datasets and gating settings"):
            st.dataframe(prefetch, hide_index=True, width="stretch")

# ------------------------------------------------------------------ #
with tabs[7]:
    st.subheader("Does the ranking survive a second dataset?")
    if robust is None:
        st.info("Run `python run_all.py --stage robustness` to generate this.")
    else:
        st.markdown(
            "An independent trace -- different seed, sharper skew, different locality target -- "
            "re-scored with the same four policies."
        )
        inv = {"Pure Static": "static", "Pure LRU": "lru",
               "Hybrid": "hybrid", "Periodic Re-profile": "periodic"}
        fig = go.Figure()
        for _, r in robust.iterrows():
            fig.add_trace(go.Scatter(
                x=["Dataset 1", "Dataset 2"],
                y=[r["dataset1_overall_ns"], r["dataset2_overall_ns"]],
                name=r["strategy"], mode="lines+markers",
                line=dict(color=S[inv[r["strategy"]]], width=3), marker=dict(size=11)))
        fig.update_layout(hovermode="closest")
        st.plotly_chart(plotly_theme(fig, ylab="Overall avg access time (ns)"), width="stretch")

        show = robust.copy()
        if "rank_change" not in show:
            show["rank_change"] = show["dataset2_rank"] - show["dataset1_rank"]
        show["Verdict"] = np.where(show["rank_change"] == 0, "Rank held",
                                   np.where(show["rank_change"] > 0,
                                            "Fell " + show["rank_change"].abs().astype(str),
                                            "Rose " + show["rank_change"].abs().astype(str)))
        st.dataframe(show, hide_index=True, width="stretch")

        held = show[show["rank_change"] == 0]["strategy"].tolist()
        st.warning(
            "**The hybrid is the one that breaks**, moving from 2nd to 4th. Sweeping its "
            "reserved/LRU split on dataset 2 shows why: the best split there reserves nothing "
            "-- that is, pure LRU. The fixed 50/50 ratio was tuned on dataset 1 and does not "
            "transfer, so we report it as a negative result rather than dropping it. "
            + (f"Held rank on both: **{', '.join(held)}**." if held else "")
        )

# ------------------------------------------------------------------ #
with tabs[8]:
    st.subheader("Method, and what it does not show")
    st.markdown(
        "The simplifications below are load-bearing. Each is a place a reviewer should "
        "discount the result."
    )
    limits = [
        ("Tier model", "latency + size/bandwidth, HBM 150 ns / 800 GB/s, CXL 220 ns / 64 GB/s. "
                       "Defensible published ballparks, not a specific part's datasheet."),
        ("No migration cost", "A miss is charged the CXL read but not the cost of installing "
                              "the expert into HBM, nor the eviction write-back. This flatters "
                              "every adaptive policy -- LRU most of all, since it migrates most "
                              "often -- and is the largest correction the model needs."),
        ("Per-token re-fetch", "Each access is charged a full expert transfer. Real inference "
                               "reads a resident expert at tile granularity without re-fetching, "
                               "so absolute latencies sit far above a real decode step. Ratios "
                               "are unaffected."),
        ("Expert size", "16 MiB is a placeholder for address-space layout. Mixtral's real "
                        "per-expert FFN is ~176 M parameters, about 336 MiB at fp16 -- roughly "
                        "21x larger."),
        ("Top-1 only", "The real-trace analysis tracks the first-choice expert. Mixtral routes "
                       "top-2, so true HBM pressure is higher and these hit rates are optimistic."),
        ("Energy constant", "21.08 nJ/fetch includes amortised background power; CXL's 3x "
                            "multiplier is a published PHY ballpark, not a measurement. Relative "
                            "savings are sound; absolute joules are not."),
        ("One model family", "Everything real is Mixtral 8x7B, 8 experts, top-2. Finer-grained "
                             "models (64-128 experts) have flatter routing distributions; only "
                             "synthetic stand-ins for those topologies were tested."),
        ("Static profiling", "run_real_benchmark.py originally profiled static placement on the "
                             "same tokens it scored -- look-ahead that inflated static's hit rate. "
                             "It now profiles on a held-out prefix; figures published before that "
                             "fix should be regenerated."),
    ]
    for term, body in limits:
        st.markdown(f"**{term}** -- {body}")

    st.info(
        "**What would change the conclusion.** Adding a migration penalty proportional to "
        "expert size would penalise LRU hardest and could plausibly reverse the layer-15 "
        "result, handing it to periodic re-profiling -- whose whole appeal is bounded migration "
        "traffic. That experiment is the obvious next one, and it is not yet run."
    )

st.divider()
st.caption(
    "CXL-Based Memory Optimisation for MoE Models | Vedant Kabra, Neil Verma | "
    "Sources: Mixtral of Experts (Jiang et al., 2024), Switch Transformer (Fedus et al., 2022), "
    "DRAMSim3 (Li et al., 2020), CXL 2.0 specification, allenai/analysis_mixtral"
)
